"""DeepAgents filesystem and shell operations inside a LexVerse environment."""
from __future__ import annotations

import asyncio
import base64
from functools import wraps
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import sys
from uuid import uuid4

from deepagents.backends.protocol import (
    EditResult, ExecuteResponse, FileDownloadResponse, FileUploadResponse, GlobResult,
    GrepResult, LsResult,
)
from deepagents.backends.sandbox import (
    BaseSandbox, _build_edit_tmpfile_cmd, _build_grep_cmd, _parse_grep_output,
)

from lexverse.environments.base import ExecutionEnvironment


def _async(method):
    @wraps(method)
    async def run(self, *args, **kwargs):
        return await asyncio.to_thread(method, self, *args, **kwargs)
    return run


class EnvironmentBackend(BaseSandbox):
    """Reuse BaseSandbox helpers; every actual file access goes through exec.

    The agent runs asynchronously. Synchronous backend hooks run in workers and
    submit environment operations to the agent's owning event loop.
    """

    def __init__(self, environment: ExecutionEnvironment, *, timeout_sec: float = 60):
        self.environment = environment
        self.timeout_sec = max(1, int(timeout_sec))
        self._loop = asyncio.get_running_loop()
        self._roots = {
            path: str(environment.map_virtual_path(path))
            for path in ("/inputs", "/skills", "/workspace", "/artifacts")
        }
        self._roots.update({
            "/" + name: str(environment.map_virtual_path("/runtime/" + name))
            for name in ("large_tool_results", "conversation_history", "blobs")
        })

    @property
    def id(self):
        return self.environment.session_id

    def _wait(self, coroutine):
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            coroutine.close()
            raise RuntimeError("Use asynchronous backend methods from the agent event loop")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def execute(self, command, *, timeout=None):
        return self._wait(self.aexecute(command, timeout=timeout))

    async def aexecute(self, command, *, timeout=None):
        timeout = self.timeout_sec if timeout is None else max(1, min(timeout, self.timeout_sec))
        env = None
        if self.environment.type().value == "local":
            # BaseSandbox helpers invoke python3; use the current interpreter.
            env = {"PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
        async with self.environment.agent_execution():
            result = await self.environment.exec(command, cwd="/workspace", timeout_sec=timeout, env=env)
        return ExecuteResponse(output=(result.stdout or "") + (result.stderr or ""),
                               exit_code=result.return_code)

    def _path(self, path):
        path = str(path)
        if ".." in PurePosixPath(path).parts or path.startswith("~"):
            raise ValueError("Path traversal is not allowed")
        if not path.startswith("/"):
            path = "/" + path
        for virtual, actual in self._roots.items():
            if path == virtual or path.startswith(virtual + "/"):
                return actual + path[len(virtual):]
        return str(self.environment.map_virtual_path(path))

    def _virtual(self, path):
        for virtual, actual in sorted(self._roots.items(), key=lambda item: -len(item[1])):
            if path == actual or path.startswith(actual + "/"):
                return virtual + path[len(actual):]
        return path

    def _result(self, result):
        if getattr(result, "path", None):
            result.path = self._virtual(result.path)
        for entry in getattr(result, "entries", None) or getattr(result, "matches", None) or []:
            entry["path"] = self._virtual(entry["path"])
        return result

    def ls(self, path):
        if path == "/":
            return LsResult(entries=[{"path": root + "/", "is_dir": True} for root in self._roots])
        return self._result(super().ls(self._path(path)))

    als = _async(ls)

    def read(self, file_path, offset=0, limit=2000):
        return super().read(self._path(file_path), offset=offset, limit=limit)

    aread = _async(read)

    def write(self, file_path, content):
        return self._result(super().write(self._path(file_path), content))

    awrite = _async(write)

    def edit(self, file_path, old_string, new_string, replace_all=False):
        return self._result(super().edit(self._path(file_path), old_string, new_string, replace_all))

    aedit = _async(edit)

    def _edit_via_upload(self, file_path, old_string, new_string, replace_all):
        # BaseSandbox's large-edit helper encodes /tmp paths in its command.
        # Translate them before encoding so Local reads its own uploaded files.
        prefix = self._path("/tmp/.deepagents_edit_" + uuid4().hex)
        old_path, new_path = prefix + "_old", prefix + "_new"
        try:
            for response in self.upload_files([(old_path, old_string.encode()), (new_path, new_string.encode())]):
                if response.error:
                    return EditResult(error=response.error)
            result = self.execute(_build_edit_tmpfile_cmd(
                file_path, old_path, new_path, replace_all=replace_all,
            ))
            if result.exit_code:
                return EditResult(error=result.output)
            data = json.loads(result.output)
            return EditResult(path=file_path if "error" not in data else None,
                              error=data.get("error"), occurrences=data.get("count"))
        finally:
            self.execute("rm -f " + shlex.quote(old_path) + " " + shlex.quote(new_path))

    def delete(self, file_path):
        return self._result(super().delete(self._path(file_path)))

    adelete = _async(delete)

    def grep(self, pattern, path=None, glob=None, *, max_count=None):
        if path in (None, "/"):
            matches = []
            truncated = False
            for root in ("/inputs", "/skills", "/workspace", "/artifacts"):
                result = self.grep(pattern, root, glob, max_count=max_count)
                if result.error:
                    return result
                matches.extend(result.matches or [])
                truncated |= result.truncated
            if max_count is not None and len(matches) > max_count:
                matches, truncated = matches[:max_count], True
            return GrepResult(matches=matches, truncated=truncated)
        if self.environment.type().value == "local" and "/" not in (glob or ""):
            # Use BaseSandbox's Python search: macOS grep -Z is not GNU's
            # NUL-filename option, which its default result parser expects.
            glob = "**/" + (glob or "*")
        if self.environment.type().value == "local":
            command = _build_grep_cmd(pattern, self._path(path), glob, max_count)
            # Quote the generated Python source as one argument. The pinned
            # helper contains unescaped double quotes in a Python comment.
            source = command[len('python3 -c "'):-len('" 2>/dev/null')]
            encoded = base64.b64encode(source.encode()).decode()
            result = self.execute("python3 -c " + shlex.quote(
                f"import base64; exec(base64.b64decode('{encoded}'))"))
            return self._result(_parse_grep_output(result, path, max_count))
        return self._result(super().grep(pattern, self._path(path), glob, max_count=max_count))

    agrep = _async(grep)

    def glob(self, pattern, path=None):
        for root in self._roots:
            if pattern.startswith(root + "/"):
                return self.glob(pattern[len(root) + 1:], root)
        if path in (None, "/"):
            matches = []
            truncated = False
            reason = None
            for root in ("/inputs", "/skills", "/workspace", "/artifacts"):
                result = self.glob(pattern.lstrip("/"), root)
                if result.error:
                    return result
                matches.extend(result.matches or [])
                truncated |= result.truncated
                reason = reason or result.truncation_reason
            return GlobResult(matches=matches, truncated=truncated, truncation_reason=reason)
        return self._result(super().glob(pattern, self._path(path)))

    aglob = _async(glob)

    def upload_files(self, files):
        responses = []
        for path, content in files:
            encoded_path = base64.b64encode(self._path(path).encode()).decode()
            error = None
            # Bound each bash -c argument, including large tool-result uploads.
            for offset in range(0, max(1, len(content)), 32768):
                chunk = base64.b64encode(content[offset:offset + 32768]).decode()
                script = ("import base64; from pathlib import Path; "
                          f"p=Path(base64.b64decode('{encoded_path}').decode()); "
                          "p.parent.mkdir(parents=True,exist_ok=True); "
                          f"p.open({'wb' if offset == 0 else 'ab'!r}).write(base64.b64decode('{chunk}'))")
                result = self.execute("python3 -c " + shlex.quote(script))
                if result.exit_code:
                    error = result.output
                    break
            responses.append(FileUploadResponse(path=path, error=error))
        return responses

    aupload_files = _async(upload_files)

    def download_files(self, paths):
        responses = []
        for path in paths:
            encoded_path = base64.b64encode(self._path(path).encode()).decode()
            script = ("import base64; from pathlib import Path; "
                      f"p=Path(base64.b64decode('{encoded_path}').decode()); "
                      "print(base64.b64encode(p.read_bytes()).decode())")
            result = self.execute("python3 -c " + shlex.quote(script))
            responses.append(FileDownloadResponse(
                path=path, content=None if result.exit_code else base64.b64decode(result.output),
                error=result.output if result.exit_code else None,
            ))
        return responses

    adownload_files = _async(download_files)

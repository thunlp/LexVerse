from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path

from lexverse.environments.base import (
    ExecutionEnvironment,
    EnvironmentCapabilities,
    ExecResult,
)
from lexverse.models.trial import EnvironmentType


class LocalEnvironment(ExecutionEnvironment):
    """
    Host-process development environment using the current user's permissions.
    It implements the Harbor-compatible execution environment contract without filesystem isolation.

    Automatically creates an isolated HOME directory under trial_paths and
    maps standard container paths (/logs, /inputs, /skills, /workspace,
    /artifacts, /tests, /solution, /installed-agent) to subdirectories of
    trial_paths.trial_dir so installed CLI agents work out of the box.
    """

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.LOCAL

    @property
    def is_mounted(self) -> bool:
        return True

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=True)

    @property
    def home_dir(self) -> Path:
        return self.trial_paths.runtime_dir / "home"

    @property
    def _virtual_mounts(self) -> dict[str, Path]:
        private_root = self.trial_paths.trial_dir
        if self.trial_paths._compact:
            private_root = private_root / ".internal"
        return {
            "/logs/agent": self.trial_paths.agent_dir,
            "/logs/verifier": self.trial_paths.verifier_dir,
            "/logs/artifacts": self.trial_paths.artifacts_dir,
            "/logs": self.trial_paths.runtime_dir / "logs",
            "/inputs": self.trial_paths.inputs_dir,
            "/skills": self.skills_root,
            "/runtime": self.trial_paths.runtime_dir,
            "/tmp": self.trial_paths.runtime_dir / "tmp",
            "/var/tmp": self.trial_paths.runtime_dir / "tmp",
            "/workspace": self.trial_paths.workspace_dir,
            "/artifacts": self.trial_paths.artifacts_dir,
            "/tests": private_root / "tests",
            "/solution": private_root / "solution",
            "/installed-agent": self.trial_paths.runtime_dir / "installed-agent",
        }

    def map_virtual_path(self, path: str) -> Path:
        posix = path.replace("\\", "/")
        for prefix, host_dir in sorted(
            self._virtual_mounts.items(), key=lambda kv: -len(kv[0])
        ):
            if posix == prefix:
                return host_dir
            if posix.startswith(prefix + "/"):
                rel = posix[len(prefix) + 1 :]
                return host_dir / rel
        p = Path(path)
        if p.is_absolute():
            return p
        return self.trial_paths.workspace_dir / p

    def _rewrite_command_paths(self, command: str) -> str:
        """Rewrite standard root-anchored container paths in shell commands to local trial paths."""
        rewritten = command
        for prefix, host_dir in sorted(
            self._virtual_mounts.items(), key=lambda kv: -len(kv[0])
        ):
            pattern = re.compile(
                rf"(^|[\s\"'=<>\(:;]){re.escape(prefix)}(?=$|[/\s\"'\):;])"
            )
            replacement_path = str(host_dir.resolve())
            rewritten = pattern.sub(lambda m, p=replacement_path: m.group(1) + p, rewritten)
        # Also rewrite relative `artifacts/` or `inputs/` paths when executed inside `/workspace`
        artifacts_host = str(self.trial_paths.artifacts_dir.resolve())
        rewritten = re.sub(
            r"(^|[\s\"'=<>\(:;])(?:\./)?artifacts(?=/|[\s\"'\):;]|$)",
            lambda m: m.group(1) + artifacts_host,
            rewritten,
        )
        return rewritten

    def _validate_definition(self) -> None:
        return None

    async def start(self, force_build: bool = False) -> None:
        if self.task_env_config.network_mode == "allowlist" or not self.task_env_config.allow_internet or self.task_env_config.network_allowlist is not None:
            raise ValueError("Local is a trusted development mode; network restrictions require Docker")
        self.trial_paths.mkdir(skills_root=self.skills_root)
        self.home_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.runtime_dir / "cache").mkdir(exist_ok=True)
        for host_dir in self._virtual_mounts.values():
            host_dir.mkdir(parents=True, exist_ok=True)

        # Seed the task HOME with symlinks to host toolchains (.nvm, .local, .cargo)
        # so pre-installed CLI agents on the host work without re-downloading Node/uv.
        host_home = Path.home()
        for tool_dir in (".nvm", ".local", ".cargo", ".npm"):
            src = host_home / tool_dir
            dst = self.home_dir / tool_dir
            if src.exists() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except OSError:
                    pass

        # Create workspace convenience symlinks (`workspace/artifacts -> ../artifacts`, `workspace/inputs -> ../inputs`)
        # so CLI agents that write to relative `artifacts/...` paths automatically land in `/artifacts/`.
        if not self.trial_paths._compact:
            for rel_name, target_dir in (
                ("artifacts", self.trial_paths.artifacts_dir),
                ("inputs", self.trial_paths.inputs_dir),
            ):
                link_path = self.trial_paths.workspace_dir / rel_name
                if not link_path.exists() and not link_path.is_symlink():
                    try:
                        link_path.symlink_to(target_dir)
                    except OSError:
                        pass
        if self.environment_dir.is_dir():
            for item in self.environment_dir.iterdir():
                if item.name in ("Dockerfile", "docker-compose.yaml"):
                    continue
                target = self.trial_paths.workspace_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                elif item.is_file():
                    shutil.copy2(item, target)

    def instruction_paths(self, instruction: str) -> str:
        return self._rewrite_command_paths(instruction)

    async def _begin_agent_execution(self):
        return None

    async def stop(self, delete: bool = True) -> None:
        return None

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        src = Path(source_path)
        dst = self.map_virtual_path(target_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
            if dst.suffix in (".sh", ".bash", ".py"):
                try:
                    dst.write_text(
                        self._rewrite_command_paths(dst.read_text(encoding="utf-8")),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        src = Path(source_dir)
        dst = self.map_virtual_path(target_dir)
        dst.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            for script in dst.rglob("*"):
                if script.is_file() and script.suffix in (".sh", ".bash", ".py"):
                    try:
                        script.write_text(
                            self._rewrite_command_paths(
                                script.read_text(encoding="utf-8")
                            ),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        src = self.map_virtual_path(source_path)
        dst = Path(target_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found in environment: {source_path}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        src = self.map_virtual_path(source_dir)
        dst = Path(target_dir)
        if not src.exists():
            return
        dst.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copytree(src, dst, dirs_exist_ok=True)

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        resolved_cwd_str = self._resolve_workdir(cwd)
        resolved_cwd = (
            self.map_virtual_path(resolved_cwd_str)
            if resolved_cwd_str
            else self.trial_paths.workspace_dir
        )
        resolved_cwd.mkdir(parents=True, exist_ok=True)

        exec_env = os.environ.copy()
        exec_env["HOME"] = str(self.home_dir.resolve())
        exec_env["LEXVERSE_WORKSPACE"] = str(self.trial_paths.workspace_dir.resolve())
        exec_env["LEXVERSE_INPUTS"] = str(self.trial_paths.inputs_dir.resolve())
        exec_env["LEXVERSE_SKILLS"] = str(self.skills_root.resolve())
        exec_env["LEXVERSE_ARTIFACTS"] = str(self.trial_paths.artifacts_dir.resolve())
        exec_env["LEXVERSE_LOGS"] = str(self.trial_paths.agent_dir.parent.resolve())

        merged_extra = self._merge_env(env)
        if merged_extra:
            for k, v in merged_extra.items():
                exec_env[k] = self._rewrite_command_paths(str(v))

        exec_env["HOME"] = str(self.home_dir.resolve())
        exec_env["TMPDIR"] = str((self.trial_paths.runtime_dir / "tmp").resolve())
        exec_env["TMP"] = exec_env["TMPDIR"]
        exec_env["TEMP"] = exec_env["TMPDIR"]
        exec_env["XDG_CACHE_HOME"] = str(self.trial_paths.runtime_dir / "cache")
        exec_env["XDG_CONFIG_HOME"] = str(self.home_dir / ".config")
        exec_env["PYTHONDONTWRITEBYTECODE"] = "1"
        rewritten_cmd = self._rewrite_command_paths(command)
        executable = ["bash", "-c", rewritten_cmd]

        process = await asyncio.create_subprocess_exec(
            *executable,
            cwd=str(resolved_cwd),
            env=exec_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            if timeout_sec is not None:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=timeout_sec
                )
            else:
                stdout_bytes, stderr_bytes = await process.communicate()
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            raise TimeoutError(f"Command timed out after {timeout_sec} seconds")

        stdout = stdout_bytes.decode(errors="replace") if stdout_bytes else None
        stderr = stderr_bytes.decode(errors="replace") if stderr_bytes else None
        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode or 0,
        )

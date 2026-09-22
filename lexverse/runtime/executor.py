"""Execute local subprocess trees in isolated Trial directories.

Timeout cleanup sends SIGTERM to the process group, waits for a grace period,
then escalates to SIGKILL. Standard streams are captured with a size limit.
"""
from __future__ import annotations

import asyncio
import os
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

from lexverse.runtime.errors import (
    ArtifactMissingError,
    TrialProcessError,
    TrialTimeoutError,
)

_STDERR_TAIL_BYTES = 8192
_TERMINATE_GRACE_SEC = 5.0


@dataclass
class ExecutionResult:
    returncode: int
    stdout_path: Path
    stderr_path: Path
    duration_sec: float
    timed_out: bool = False
    artifacts: dict[str, Path] = field(default_factory=dict)


async def run_subprocess(
    command: list[str],
    work_dir: Path,
    timeout_sec: float,
    env: dict[str, str] | None = None,
    expected_artifacts: dict[str, str] | None = None,
) -> ExecutionResult:
    """Run a subprocess in an isolated work directory with a timeout.

    Arguments:
        command: argv list; not passed through a shell.
        work_dir: created if missing; used as the subprocess cwd.
        timeout_sec: wall-clock budget; SIGTERM then SIGKILL on breach.
        env: full environment for the child (None -> inherit parent).
        expected_artifacts: name -> path relative to work_dir; raises
            ArtifactMissingError if the subprocess succeeds but a listed
            path is missing.

    Returns:
        ExecutionResult with returncode, log paths, duration, timeout flag,
        and resolved artifact paths.

    Raises:
        TrialTimeoutError: subprocess killed after grace period.
        TrialProcessError: non-zero exit that is not a timeout.
        ArtifactMissingError: exit 0 but a required output is absent.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = work_dir / "stdout.log"
    stderr_path = work_dir / "stderr.log"

    start = time.monotonic()
    with stdout_path.open("wb") as stdout_fp, stderr_path.open("wb") as stderr_fp:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(work_dir),
            env=env,
            stdout=stdout_fp,
            stderr=stderr_fp,
            start_new_session=True,  # own process group for clean shutdown
        )

        timed_out = False
        try:
            returncode = await asyncio.wait_for(proc.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            timed_out = True
            returncode = await _terminate(proc)

    duration_sec = time.monotonic() - start

    if timed_out:
        raise TrialTimeoutError(
            f"trial exceeded {timeout_sec}s (killed with returncode={returncode})"
        )

    if returncode != 0:
        raise TrialProcessError(returncode, _tail(stderr_path, _STDERR_TAIL_BYTES))

    resolved = _resolve_artifacts(work_dir, expected_artifacts or {})

    return ExecutionResult(
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        duration_sec=duration_sec,
        timed_out=False,
        artifacts=resolved,
    )


async def _terminate(proc: asyncio.subprocess.Process) -> int:
    """SIGTERM the process group, then SIGKILL after a grace period."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return await proc.wait()

    try:
        return await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_GRACE_SEC)
    except asyncio.TimeoutError:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return await proc.wait()


def _tail(path: Path, nbytes: int) -> str:
    try:
        with path.open("rb") as fp:
            fp.seek(0, os.SEEK_END)
            size = fp.tell()
            fp.seek(max(0, size - nbytes))
            return fp.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _resolve_artifacts(work_dir: Path, expected: dict[str, str]) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name, rel in expected.items():
        candidate = work_dir / rel
        if not candidate.exists():
            raise ArtifactMissingError(f"{name} -> {rel}")
        resolved[name] = candidate
    return resolved

from __future__ import annotations

import asyncio
import hashlib
import shlex
from pathlib import Path

from lexverse.environments.base import (
    BaseEnvironment,
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
    ExecResult,
)
from lexverse.models.trial import EnvironmentPaths, EnvironmentType


class DockerEnvironment(BaseEnvironment):
    """
    Harbor-compatible Docker container sandbox environment.

    Supports prebuilt images (`task_env_config.docker_image`), custom
    `environment/Dockerfile`, or a default `python:3.12-slim` image.
    Automatically bind-mounts `/inputs` (ro), `/skills` (ro), `/workspace` (rw),
    `/artifacts` (rw), `/logs/agent` (rw), and `/logs/verifier` (rw).
    """

    DEFAULT_IMAGE = "python:3.12-slim"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import re

        safe_session = re.sub(r"[^a-z0-9_.-]+", "-", self.session_id.lower()).strip("-")
        safe_env_name = re.sub(r"[^a-z0-9_.-]+", "-", self.environment_name.lower()).strip("-")
        self._container_name = f"lexverse-{safe_session[:48]}"
        self._image_name = (
            self.task_env_config.docker_image
            or f"lexverse-env__{safe_env_name[:48]}:latest"
        )

    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER

    @property
    def is_mounted(self) -> bool:
        return True

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(
            gpus=True,
            disable_internet=True,
            mounted=True,
        )

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(
            cpu_limit=True,
            memory_limit=True,
        )

    def _validate_definition(self) -> None:
        return None

    async def _run_docker(
        self, args: list[str], timeout_sec: int | None = None
    ) -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
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
            raise TimeoutError(f"Docker command timed out after {timeout_sec} seconds")

        return ExecResult(
            stdout=stdout_bytes.decode(errors="replace") if stdout_bytes else None,
            stderr=stderr_bytes.decode(errors="replace") if stderr_bytes else None,
            return_code=process.returncode or 0,
        )

    async def start(self, force_build: bool = False) -> None:
        self.trial_paths.mkdir()
        self.trial_paths.chmod_dir()

        dockerfile = self.environment_dir / "Dockerfile"
        if dockerfile.exists() and (force_build or not self.task_env_config.docker_image):
            df_hash = hashlib.sha256(dockerfile.read_bytes()).hexdigest()[:12]
            self._image_name = f"lexverse-env__df-{df_hash}:latest"
            inspect_res = await self._run_docker(["image", "inspect", self._image_name])
            if force_build or inspect_res.return_code != 0:
                build_args = [
                    "build",
                    "-t",
                    self._image_name,
                    "-f",
                    str(dockerfile.resolve()),
                    str(self.environment_dir.resolve()),
                ]
                res = await self._run_docker(
                    build_args, timeout_sec=int(self.task_env_config.build_timeout_sec)
                )
                if res.return_code != 0:
                    raise RuntimeError(
                        f"Docker image build failed (exit {res.return_code}): {res.stderr or res.stdout}"
                    )
        elif not self.task_env_config.docker_image:
            self._image_name = self.DEFAULT_IMAGE

        # Remove any stale container with the same name
        await self._run_docker(["rm", "-f", self._container_name])

        repo_root = Path(__file__).resolve().parents[2]
        cap_cache_dir = repo_root / ".lexverse/capabilities"
        cap_cache_dir.mkdir(parents=True, exist_ok=True)
        internal_dir = self.trial_paths.trial_dir / ".internal"
        internal_dir.mkdir(parents=True, exist_ok=True)

        workdir = self.task_env_config.workdir or str(EnvironmentPaths.workspace_dir)
        run_args = [
            "run",
            "-d",
            "--name",
            self._container_name,
            "-w",
            workdir,
            "--cpus",
            str(self.task_env_config.cpus),
            "--memory",
            f"{self.task_env_config.memory_mb}m",
            "-v",
            f"{self.trial_paths.inputs_dir.resolve()}:/inputs:ro",
            "-v",
            f"{self.trial_paths.skills_dir.resolve()}:/skills:ro",
            "-v",
            f"{self.trial_paths.skills_dir.resolve()}:/harbor/skills:ro",
            "-v",
            f"{self.trial_paths.workspace_dir.resolve()}:/workspace:rw",
            "-v",
            f"{self.trial_paths.artifacts_dir.resolve()}:/artifacts:rw",
            "-v",
            f"{self.trial_paths.artifacts_dir.resolve()}:/logs/artifacts:rw",
            "-v",
            f"{self.trial_paths.agent_dir.resolve()}:/logs/agent:rw",
            "-v",
            f"{self.trial_paths.verifier_dir.resolve()}:/logs/verifier:rw",
            "-v",
            f"{repo_root}:/lexverse-repo:ro",
            "-v",
            f"{cap_cache_dir.resolve()}:/lexverse-cache:ro",
            "-v",
            f"{cap_cache_dir.resolve()}:{cap_cache_dir.resolve()}:ro",
            "-v",
            f"{internal_dir.resolve()}:/lexverse-internal:ro",
            "-v",
            f"{internal_dir.resolve()}:{internal_dir.resolve()}:ro",
        ]

        # Mount pre-cached local embedding models into the container if present
        bge_dir = repo_root / ".lexverse/models"
        if bge_dir.is_dir():
            run_args.extend(["-v", f"{bge_dir.resolve()}:{bge_dir.resolve()}:ro"])

        # Mount host toolchains (.nvm, .local, .codex) at their host paths read-only
        # so host symlinks work, without shadowing container /root/.local or /root/.nvm.
        host_home = Path.home()
        for host_sub in (".nvm", ".local", ".codex/packages"):
            sub_path = host_home / host_sub
            if sub_path.is_dir():
                run_args.extend(["-v", f"{sub_path.resolve()}:{sub_path.resolve()}:ro"])

        if not self.task_env_config.allow_internet:
            run_args.extend(["--network", "none"])
        else:
            run_args.extend(["--add-host", "host.docker.internal:host-gateway"])

        if self.task_env_config.gpus > 0:
            run_args.extend(["--gpus", f"device={self.task_env_config.gpus}"])

        run_args.extend(
            [self._image_name, "sh", "-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
        )

        res = await self._run_docker(run_args, timeout_sec=180)
        if res.return_code != 0:
            raise RuntimeError(
                f"Failed to start Docker container {self._container_name}: {res.stderr or res.stdout}"
            )
        if not self.trial_paths._compact:
            await self.exec(
                "ln -sfn /artifacts /workspace/artifacts && "
                "ln -sfn /artifacts /workspace/output && "
                "ln -sfn /inputs /workspace/inputs && "
                "ln -sfn /inputs /workspace/documents && "
                "mkdir -p /workspace/skills /root/.agents/skills && "
                "for f in $(find /skills -name 'SKILL.md' 2>/dev/null); do "
                "  d=$(dirname \"$f\"); "
                "  ln -sfn \"$d\" \"/workspace/skills/$(basename \"$d\")\" || true; "
                "done",
                user="root",
            )

    async def stop(self, delete: bool = True) -> None:
        if self.keep_containers and not delete:
            return
        try:
            import os

            uid, gid = os.getuid(), os.getgid()
            await self.exec(
                f"chmod -R u+rwX /workspace /artifacts /logs/agent /logs/verifier 2>/dev/null || true; "
                f"chown -hR {uid}:{gid} /workspace /artifacts /logs/agent /logs/verifier 2>/dev/null || true",
                user="root",
                timeout_sec=15,
            )
        except Exception:
            pass
        await self._run_docker(["rm", "-f", self._container_name], timeout_sec=60)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        parent = str(Path(target_path).parent)
        await self.exec(f"mkdir -p {shlex.quote(parent)}", user="root")
        res = await self._run_docker(
            ["cp", str(Path(source_path).resolve()), f"{self._container_name}:{target_path}"]
        )
        if res.return_code != 0:
            raise RuntimeError(f"Failed to upload file to container: {res.stderr}")

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self.exec(f"mkdir -p {shlex.quote(target_dir)}", user="root")
        res = await self._run_docker(
            ["cp", f"{Path(source_dir).resolve()}/.", f"{self._container_name}:{target_dir}"]
        )
        if res.return_code != 0:
            raise RuntimeError(f"Failed to upload directory to container: {res.stderr}")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        dst = Path(target_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        res = await self._run_docker(
            ["cp", f"{self._container_name}:{source_path}", str(dst.resolve())]
        )
        if res.return_code != 0:
            raise RuntimeError(f"Failed to download file from container: {res.stderr}")

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        dst = Path(target_dir)
        dst.mkdir(parents=True, exist_ok=True)
        res = await self._run_docker(
            ["cp", f"{self._container_name}:{source_dir}/.", str(dst.resolve())]
        )
        if res.return_code != 0:
            raise RuntimeError(f"Failed to download directory from container: {res.stderr}")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        args = ["exec"]
        workdir = cwd or self.task_env_config.workdir or "/workspace"
        args.extend(["-w", workdir])
        args.extend(["-e", "PYTHONPATH=/lexverse-repo"])
        extra_paths = ["/root/.local/bin"]
        fnm_dir = Path.home() / ".local/share/fnm/node-versions"
        if fnm_dir.is_dir():
            for vdir in sorted(fnm_dir.glob("*/installation/bin"), reverse=True):
                extra_paths.append(str(vdir.resolve()))
        args.extend([
            "-e",
            f"PATH={':'.join(extra_paths)}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        ])

        merged_env = self._merge_env(env)
        if merged_env:
            for k, v in merged_env.items():
                val = str(v)
                if k in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "LLM_BASE_URL", "ANTHROPIC_BASE_URL"):
                    val = val.replace("://127.0.0.1:", "://host.docker.internal:").replace("://localhost:", "://host.docker.internal:")
                args.extend(["-e", f"{k}={val}"])

        resolved_user = self._resolve_user(user)
        if resolved_user is not None:
            args.extend(["-u", str(resolved_user)])

        args.extend([self._container_name, "bash", "-c", command])
        return await self._run_docker(args, timeout_sec=timeout_sec)

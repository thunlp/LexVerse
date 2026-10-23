from __future__ import annotations

import asyncio
import hashlib
import shlex
from pathlib import Path

from lexverse.environments.base import (
    ExecutionEnvironment,
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
    ExecResult,
)
from lexverse.models.trial import EnvironmentPaths, EnvironmentType
from lexverse.environments.permissions import READ_ONLY, READ_WRITE, RUNTIME, validate_roots
from lexverse.environments.network import NetworkPolicy


class DockerEnvironment(ExecutionEnvironment):
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
        self._prepared_container = self._container_name
        self._execution_image = None
        self._network_container = self._prepared_container + "-network"
        self._network_monitor = None
        self._network_failed = False
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
            network_allowlist=True,
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
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
            await process.communicate()
            raise

        return ExecResult(
            stdout=stdout_bytes.decode(errors="replace") if stdout_bytes else None,
            stderr=stderr_bytes.decode(errors="replace") if stderr_bytes else None,
            return_code=process.returncode or 0,
        )

    async def start(self, force_build: bool = False) -> None:
        if self.task_env_config.network_allowlist is not None:
            raise ValueError("Use network_mode='allowlist' and allowed_hosts instead of network_allowlist")
        self._network_policy = NetworkPolicy(network_mode=self.task_env_config.network_mode,
                                             allowed_hosts=self.task_env_config.allowed_hosts)
        self.trial_paths.mkdir(skills_root=self.skills_root)
        self.trial_paths.chmod_dir()
        for name in ("home", "tmp", "cache", "installed-agent", "shm"):
            (self.trial_paths.runtime_dir / name).mkdir(parents=True, exist_ok=True)

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
            f"{self.trial_paths.inputs_dir.resolve()}:{READ_ONLY[0]}:ro",
            "-v",
            f"{self.skills_root.resolve()}:{READ_ONLY[1]}:ro",
            "-v",
            f"{self.trial_paths.runtime_dir.resolve()}:{RUNTIME}:rw",
            "-v",
            f"{(self.trial_paths.runtime_dir / 'installed-agent').resolve()}:/installed-agent:rw",
            "-v",
            f"{(self.trial_paths.runtime_dir / 'tmp').resolve()}:/tmp:rw",
            "-v",
            f"{(self.trial_paths.runtime_dir / 'tmp').resolve()}:/var/tmp:rw",
            "-v",
            f"{(self.trial_paths.runtime_dir / 'shm').resolve()}:/dev/shm:rw",
            "-v",
            f"{self.trial_paths.workspace_dir.resolve()}:{READ_WRITE[0]}:rw",
            "-v",
            f"{self.trial_paths.artifacts_dir.resolve()}:{READ_WRITE[1]}:rw",
            "-v",
            f"{self.trial_paths.artifacts_dir.resolve()}:/logs/artifacts:rw",
            "-v",
            f"{self.trial_paths.agent_dir.resolve()}:/logs/agent:rw",
            "-v",
            f"{self.trial_paths.verifier_dir.resolve()}:/logs/verifier:rw",
            "-v",
            f"{repo_root / 'lexverse'}:/lexverse-repo/lexverse:ro",
            "-v",
            f"{repo_root / '.lexverse/runtime/agent'}:/lexverse-repo/.lexverse/runtime/agent:ro",
            "-v",
            f"{cap_cache_dir.resolve()}:/lexverse-cache:ro",
            "-v",
            f"{cap_cache_dir.resolve()}:{cap_cache_dir.resolve()}:ro",
            "-v",
            f"{internal_dir.resolve()}:/lexverse-internal:ro",
            "-v",
            f"{internal_dir.resolve()}:{internal_dir.resolve()}:ro",
        ]

        for path in self.knowledge_read_paths():
            run_args.extend(["-v", f"{path.resolve()}:{path.resolve()}:ro"])
        models = cap_cache_dir / "cache/models"
        if models.is_dir():
            run_args.extend(["-v", f"{models.resolve()}:{models.resolve()}:ro"])

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
        self._container_args = run_args[:-4]

    async def _begin_agent_execution(self):
        self._network_failed = False
        validate_roots([self.trial_paths.inputs_dir, self.skills_root],
                       [self.trial_paths.workspace_dir, self.trial_paths.artifacts_dir, self.trial_paths.runtime_dir])
        # Setup is trusted; freeze its installed dependencies into a disposable image.
        # Bind mounts (including Skills, credentials and task data) are not copied by docker commit.
        result = await self.exec(
            "chown -R 1000:1000 /runtime /workspace /artifacts && "
            "chmod a+rX /root && if [ -d /installed-agent ]; then chmod -R a+rX /installed-agent; fi", user="root")
        if result.return_code:
            raise RuntimeError("Cannot prepare agent runtime permissions: " + (result.stderr or ""))
        self._execution_image = self._prepared_container + "-execution:latest"
        result = await self._run_docker(["commit", self._prepared_container, self._execution_image], timeout_sec=180)
        if result.return_code:
            raise RuntimeError("Cannot prepare read-only agent image: " + (result.stderr or ""))
        self._container_name = self._prepared_container + "-agent"
        args = list(self._container_args)
        args[args.index("--name") + 1] = self._container_name
        if self._network_policy.network_mode == "allowlist":
            await self._start_network_control()
            if "--network" in args:
                n = args.index("--network")
                del args[n:n + 2]
            if "--add-host" in args:
                n = args.index("--add-host")
                del args[n:n + 2]
            args += ["--network", "container:" + self._network_container]
            hosts = self.trial_paths.trial_dir / ".internal/network-hosts"
            result = await self._run_docker(["exec", self._network_container, "cat", "/etc/hosts"])
            if result.return_code:
                raise RuntimeError("Cannot prepare task network hosts")
            hosts.write_text(result.stdout)
        # Management files and verifier logs are absent from the agent container.
        for n in range(len(args) - 2, -1, -1):
            if args[n] == "-v" and (args[n + 1].startswith(str((self.trial_paths.trial_dir / ".internal").resolve()) + ":")
                                    or ":/logs/verifier:" in args[n + 1]
                                    or args[n + 1].startswith(str((Path(__file__).resolve().parents[2] / ".lexverse/capabilities").resolve()) + ":")):
                del args[n:n + 2]
        if self._network_policy.network_mode == "allowlist":
            args += ["-v", f"{hosts.resolve()}:/etc/hosts:ro"]
        args += ["--read-only", "--cap-drop=ALL", "--security-opt", "no-new-privileges", "--user", "1000:1000",
                 "-e", "HOME=/runtime/home", "-e", "TMPDIR=/runtime/tmp", self._execution_image,
                 "sh", "-c", "trap 'exit 0' TERM; while :; do sleep 1; done"]
        result = await self._run_docker(args, timeout_sec=180)
        if result.return_code:
            self._container_name = self._prepared_container
            raise RuntimeError("Cannot start isolated agent container: " + (result.stderr or ""))
        if self._network_policy.network_mode == "allowlist":
            self._network_monitor = asyncio.create_task(self._monitor_network())

    async def _start_network_control(self):
        context = Path(__file__).with_name("egress")
        digest = hashlib.sha256(b"".join(path.read_bytes() for path in sorted(context.rglob("*")) if path.is_file())).hexdigest()[:16]
        image = "lexverse-egress:" + digest
        result = await self._run_docker(["image", "inspect", image])
        if result.return_code:
            result = await self._run_docker(["build", "-t", image, str(context)], timeout_sec=600)
            if result.return_code:
                raise RuntimeError("Cannot build task network controller: " + (result.stderr or ""))
        await self._run_docker(["rm", "-f", self._network_container])
        result = await self._run_docker([
            "run", "-d", "--name", self._network_container,
            "--cap-drop=ALL", "--cap-add=NET_ADMIN", "--cap-add=NET_RAW",
            "--security-opt", "no-new-privileges",
            "--add-host", "host.docker.internal:host-gateway",
            "-e", "EGRESS_CONTROL_INITIAL_NETWORK_MODE=allowlist",
            "-e", "EGRESS_CONTROL_INITIAL_ALLOWED_HOSTS=" + " ".join(self._network_policy.allowed_hosts),
            "--entrypoint", "/opt/egress-sidecar/entrypoint.sh", image,
        ], timeout_sec=180)
        if result.return_code:
            raise RuntimeError("Cannot start task network controller: " + (result.stderr or ""))
        for _ in range(60):
            result = await self._run_docker(["exec", self._network_container, "test", "-f", "/tmp/harbor-docker-egress-control-sidecar.ready"])
            if result.return_code == 0:
                return
            state = await self._run_docker(["inspect", "-f", "{{.State.Running}}", self._network_container])
            if (state.stdout or "").strip() != "true":
                break
            await asyncio.sleep(0.5)
        logs = await self._run_docker(["logs", self._network_container])
        raise RuntimeError("Docker cannot enforce allowlist networking: " + (logs.stderr or logs.stdout or "controller not ready"))

    async def _monitor_network(self):
        await self._run_docker(["wait", self._network_container])
        self._network_failed = True
        await self._run_docker(["kill", self._prepared_container + "-agent"])

    async def _end_agent_execution(self):
        agent_container = self._container_name
        self._container_name = self._prepared_container
        await self._run_docker(["rm", "-f", agent_container])
        await self._stop_network_control()
        if self._network_failed:
            raise RuntimeError("Task network controller stopped during execution")

    async def _stop_network_control(self):
        if self._network_monitor is not None:
            self._network_monitor.cancel()
            try:
                await self._network_monitor
            except asyncio.CancelledError:
                pass
            self._network_monitor = None
        if self._network_policy and self._network_policy.network_mode == "allowlist":
            await self._run_docker(["rm", "-f", self._network_container])

    async def stop(self, delete: bool = True) -> None:
        if self.keep_containers and not delete:
            return
        await self._run_docker(["rm", "-f", self._prepared_container + "-agent"])
        await self._stop_network_control()
        try:
            import os

            uid, gid = os.getuid(), os.getgid()
            await self.exec(
                f"chmod -R u+rwX /workspace /artifacts /logs/agent /logs/verifier 2>/dev/null || true; "
                f"chown -hR {uid}:{gid} /workspace /artifacts /runtime /logs/verifier 2>/dev/null || true",
                user="root",
                timeout_sec=15,
            )
        except Exception:
            pass
        await self._run_docker(["rm", "-f", self._container_name], timeout_sec=60)
        if self._execution_image:
            await self._run_docker(["image", "rm", "-f", self._execution_image], timeout_sec=60)

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
        args.extend(["-e", "PYTHONPATH=/lexverse-repo", "-e", "HOME=/runtime/home",
                     "-e", "TMPDIR=/runtime/tmp", "-e", "XDG_CACHE_HOME=/runtime/cache",
                     "-e", "PYTHONDONTWRITEBYTECODE=1"])
        extra_paths = ["/runtime/home/.local/bin", "/runtime/home/.npm-global/bin", "/root/.local/bin"]
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

        if self._execution_depth:
            args.extend(["-e", "HOME=/runtime/home", "-e", "TMPDIR=/runtime/tmp"])
        resolved_user = "1000:1000" if self._execution_depth else self._resolve_user(user)
        if resolved_user is not None:
            args.extend(["-u", str(resolved_user)])

        args.extend([self._container_name, "bash", "-c", command])
        return await self._run_docker(args, timeout_sec=timeout_sec)

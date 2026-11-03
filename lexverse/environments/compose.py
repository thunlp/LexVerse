import copy
from pathlib import Path
import re
import shlex
from urllib.parse import urlsplit, urlunsplit

import yaml

from lexverse.environments.base import EnvironmentCapabilities
from lexverse.environments.docker import DockerEnvironment


class ComposeEnvironment(DockerEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._project = "lexverse-" + re.sub(r"[^a-z0-9_-]", "-", self.session_id.lower())
        self._files = []
        self._service = "main"
        self._services = set()

    @property
    def capabilities(self):
        return EnvironmentCapabilities(mounted=True, docker_compose=True)

    async def _compose(self, args, timeout_sec=None):
        command = ["compose", "--project-name", self._project,
                   "--project-directory", str(self.environment_dir.resolve())]
        for path in self._files:
            command += ["-f", str(path)]
        return await self._run_docker([*command, *args], timeout_sec=timeout_sec)

    async def start(self, force_build=False):
        if self.task_env_config.network_mode != "public" or not self.task_env_config.allow_internet:
            raise ValueError("Compose does not support network restrictions; use the single-container Docker backend")
        overlay = self.environment_dir / "docker-compose.yaml"
        definition = yaml.safe_load(overlay.read_text())
        services = definition.get("services", {}) if isinstance(definition, dict) else {}
        if not isinstance(services, dict):
            raise ValueError("Compose services must be a mapping")
        if any("container_name" in service for service in services.values()):
            raise ValueError("Compose container_name would prevent per-Trial isolation")
        for section in ("volumes", "networks"):
            for resource in (definition.get(section) or {}).values():
                if resource and resource.get("name") and not resource.get("external"):
                    raise ValueError("Explicit shared resource names require external: true")
        self._services = {"main", *services}
        self.trial_paths.mkdir(skills_root=self.skills_root)
        runtime = self.trial_paths.runtime_dir.resolve()
        for directory in ("home", "tmp", "cache", "installed-agent"):
            (runtime / directory).mkdir(exist_ok=True)
        main = {
            "command": ["sh", "-c", "trap 'exit 0' TERM; while :; do sleep 1; done"],
            "working_dir": self.task_env_config.workdir or "/workspace",
            "environment": self.task_env_config.env,
            "cpus": self.task_env_config.cpus,
            "mem_limit": f"{self.task_env_config.memory_mb}m",
            "volumes": [
                f"{runtime}:/runtime",
                f"{self.trial_paths.workspace_dir.resolve()}:/workspace",
                f"{self.trial_paths.artifacts_dir.resolve()}:/artifacts",
                f"{self.trial_paths.agent_dir.resolve()}:/logs/agent",
                f"{self.trial_paths.verifier_dir.resolve()}:/logs/verifier",
                f"{self.skills_root.resolve()}:/skills:ro",
                f"{Path(__file__).resolve().parents[2]}:/lexverse-repo:ro",
            ],
        }
        if self.task_env_config.docker_image:
            main["image"] = self.task_env_config.docker_image
        elif not {"image", "build"}.intersection(services.get("main", {})):
            if (self.environment_dir / "Dockerfile").is_file():
                main["build"] = {"context": str(self.environment_dir.resolve())}
            else:
                main["image"] = self.DEFAULT_IMAGE
        base = runtime / "compose.yaml"
        base.write_text(yaml.safe_dump({"services": {"main": main}}, sort_keys=False))
        self._files = [base, overlay.resolve()]
        published = {}
        for server in self.task_env_config.mcp_servers:
            if not server.url:
                continue
            endpoint = urlsplit(server.url)
            if endpoint.hostname not in self._services:
                continue
            port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
            existing = services.get(endpoint.hostname, {}).get("ports", [])
            exposed = False
            for entry in existing:
                if isinstance(entry, dict):
                    target, protocol = str(entry["target"]), entry.get("protocol", "tcp")
                else:
                    target, _, protocol = str(entry).rsplit(":", 1)[-1].partition("/")
                    protocol = protocol or "tcp"
                bounds = target.split("-", 1)
                if protocol == "tcp" and int(bounds[0]) <= port <= int(bounds[-1]):
                    exposed = True
                    break
            if not exposed:
                ports = published.setdefault(endpoint.hostname, {"ports": []})["ports"]
                mapping = {"target": port, "published": "0", "host_ip": "127.0.0.1"}
                if mapping not in ports:
                    ports.append(mapping)
        if published:
            forwarding = runtime / "mcp-ports.yaml"
            forwarding.write_text(yaml.safe_dump({"services": published}))
            self._files.append(forwarding)
        if force_build:
            result = await self._compose(["build", "--no-cache"], int(self.task_env_config.build_timeout_sec))
            if result.return_code:
                raise RuntimeError(f"Compose build failed: {result.stderr or result.stdout}")
        result = await self._compose(["up", "-d", "--build", "--wait"], int(self.task_env_config.build_timeout_sec))
        if result.return_code:
            raise RuntimeError(f"Compose startup failed: {result.stderr or result.stdout}")

    def for_service(self, service):
        if service not in self._services:
            raise ValueError(f"Unknown Compose service: {service}")
        selected = copy.copy(self)
        selected._service = service
        return selected

    async def _begin_agent_execution(self):
        result = await self.exec("chown -R 1000:1000 /runtime /workspace /artifacts", user="root")
        if result.return_code:
            raise RuntimeError("Cannot prepare Compose Agent permissions")

    async def _end_agent_execution(self):
        pass

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        args = ["exec", "-T"]
        selected_user = "1000:1000" if self._execution_depth else self._resolve_user(user)
        if selected_user is not None:
            args += ["--user", str(selected_user)]
        workdir = cwd or (self.task_env_config.workdir if self._service == "main" else None)
        if workdir:
            args += ["--workdir", workdir]
        merged = self._merge_env(env) or {}
        variables = {"HOME": "/runtime/home", "TMPDIR": "/runtime/tmp", "PYTHONPATH": "/lexverse-repo",
                     **merged} if self._service == "main" else merged
        for key, value in variables.items():
            args += ["--env", f"{key}={value}"]
        return await self._compose([*args, self._service, "sh", "-c", command], timeout_sec)

    async def upload_file(self, source_path, target_path):
        await self.exec(f"mkdir -p {shlex.quote(str(Path(target_path).parent))}", user="root")
        result = await self._compose(["cp", str(Path(source_path).resolve()), f"{self._service}:{target_path}"])
        if result.return_code:
            raise RuntimeError(f"Compose upload failed: {result.stderr}")

    async def upload_dir(self, source_dir, target_dir):
        await self.exec(f"mkdir -p {shlex.quote(target_dir)}", user="root")
        result = await self._compose(["cp", f"{Path(source_dir).resolve()}/.", f"{self._service}:{target_dir}"])
        if result.return_code:
            raise RuntimeError(f"Compose upload failed: {result.stderr}")

    async def download_file(self, source_path, target_path):
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        result = await self._compose(["cp", f"{self._service}:{source_path}", str(target_path)])
        if result.return_code:
            raise RuntimeError(f"Compose download failed: {result.stderr}")

    async def download_dir(self, source_dir, target_dir):
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        result = await self._compose(["cp", f"{self._service}:{source_dir}/.", str(target_dir)])
        if result.return_code:
            raise RuntimeError(f"Compose download failed: {result.stderr}")

    async def download_task_skills(self, source, destination):
        if source == "/skills" or source.startswith("/skills/"):
            result = await self._compose(["images", "--quiet", "main"])
            if result.return_code or not (result.stdout or "").strip():
                raise RuntimeError("Cannot resolve the Compose main image")
            self._image_name = result.stdout.strip().splitlines()[0]
        await super().download_task_skills(source, destination)

    async def mcp_for_host(self, server):
        if server.transport == "stdio":
            args = ["compose", "--project-name", self._project, "--project-directory", str(self.environment_dir.resolve())]
            for path in self._files:
                args += ["-f", str(path)]
            args += ["exec", "-T"]
            for name, value in server.env.items():
                args += ["--env", f"{name}={value}"]
            return server.model_copy(update={"command": "docker", "args": [*args, "main", server.command, *server.args], "env": {}})
        endpoint = urlsplit(server.url)
        if endpoint.hostname not in self._services:
            return server
        port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
        result = await self._compose(["port", endpoint.hostname, str(port)])
        if result.return_code or not (result.stdout or "").strip():
            raise RuntimeError(f"Compose MCP service port is not published: {server.name}")
        host_port = result.stdout.strip().splitlines()[0].rsplit(":", 1)[1]
        return server.model_copy(update={"url": urlunsplit(endpoint._replace(netloc="127.0.0.1:" + host_port))})

    async def stop(self, delete=True):
        if not self._files or (self.keep_containers and not delete):
            return
        try:
            logs = await self._compose(["logs", "--no-color"], 30)
            (self.trial_paths.runtime_dir / "compose.log").write_text((logs.stdout or "") + (logs.stderr or ""))
        finally:
            result = await self._compose(["down", "--volumes", "--remove-orphans"], 60)
            if result.return_code:
                raise RuntimeError(f"Compose cleanup failed: {result.stderr}")

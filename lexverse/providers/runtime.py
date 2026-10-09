from __future__ import annotations

import hashlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from urllib.request import urlopen

from lexverse.config import ConfigError, ModelConfig, load_profile
from lexverse.runtime.results import atomic_write_json
from .openai_compatible import OpenAICompatibleProvider


def loading_identity(model: ModelConfig) -> str:
    value = {"provider": model.provider, "name": model.name, "load_parameters": model.load_parameters, "input": model.input}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


class LocalProvider(OpenAICompatibleProvider):
    def __init__(self, *, runtime, **kwargs):
        super().__init__(**kwargs)
        self.runtime = runtime

    async def generate(self, messages, *, config=None):
        try:
            return await super().generate(messages, config=config)
        except BaseException as exc:
            # A rejected request does not invalidate the loaded model or connection.
            if getattr(exc.__cause__, "status_code", None) != 400:
                self.runtime.stop_worker()
            raise


def scoring_identity(provider) -> dict:
    """Local model aliases identify loading settings independently of the worker port."""
    return {"model": getattr(provider, "model", None),
            "endpoint_sha256": None if isinstance(provider, LocalProvider) else hashlib.sha256(
                (getattr(provider, "base_url", None) or "").encode()).hexdigest(),
            "parameters": getattr(provider, "default_config", {})}


class ModelRuntime:
    def __init__(self, run_dir: Path, *, startup_timeout: float = 1800, request_timeout: float | None = None):
        self.run_dir = Path(run_dir)
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.process = None
        self.worker_identity = None
        self.worker_info = None
        self.providers = []
        self.manifest = None
        self.provenance_path = self.run_dir / "models.json"
        self.provenance = json.loads(self.provenance_path.read_text()) if self.provenance_path.exists() else {}

    def bind_manifest(self, manifest: dict):
        self.manifest = manifest

    def validate_phase(self, models: list[ModelConfig]) -> None:
        local = {loading_identity(m) for m in models if m.provider != "openai_compatible"}
        if len(local) > 1:
            raise ConfigError("only one distinct local model loading/input configuration is allowed per phase")

    def provider(self, model: ModelConfig, *, label: str) -> OpenAICompatibleProvider:
        identity = loading_identity(model)
        previous = self.provenance.get(label)
        template = model.input.get("chat_template")
        template_hash = hashlib.sha256(Path(template).read_bytes()).hexdigest() if template else None
        if previous and previous["identity"] != identity:
            raise ConfigError(f"model configuration changed for {label}; start a new run")
        if previous and previous.get("template_sha256") != template_hash:
            raise ConfigError(f"chat template changed for {label}; start a new run")
        if model.provider == "openai_compatible":
            profile = load_profile(model.profile)
            provider = OpenAICompatibleProvider(
                name=profile.name, model=model.name, api_key=profile.api_key, base_url=profile.base_url,
                default_config=model.generation_parameters, timeout_sec=self.request_timeout,
            )
            record = {"identity": identity, "config": asdict(model)}
        else:
            if self.worker_identity != identity:
                self.stop_worker()
                self._start_worker(model, previous)
            if self.process is None or self.process.poll() is not None:
                raise ConfigError("local inference process is no longer running")
            provider = LocalProvider(
                runtime=self, name=model.provider, model=self.alias, api_key=self.api_key,
                base_url=f"http://127.0.0.1:{self.worker_info['port']}/v1",
                default_config=model.generation_parameters, timeout_sec=self.request_timeout, max_retries=0,
            )
            record = {"identity": identity, "config": asdict(model),
                      **{k: v for k, v in self.worker_info.items() if k != "port"}}
        record["template_sha256"] = template_hash
        self.provenance[label] = record
        atomic_write_json(self.provenance_path, self.provenance)
        if self.manifest is not None:
            self.manifest["models_sha256"] = hashlib.sha256(self.provenance_path.read_bytes()).hexdigest()
            atomic_write_json(self.run_dir / "manifest.json", {k: v for k, v in self.manifest.items() if k != "bundle"})
        self.providers.append(provider)
        return provider

    async def close(self):
        self.stop_worker()
        for provider in self.providers:
            client = getattr(provider, "_client", None)
            if client is not None:
                await client.close()
        self.providers.clear()

    def stop_worker(self):
        proc, self.process = self.process, None
        self.worker_identity = self.worker_info = None
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    def _start_worker(self, model: ModelConfig, previous: dict | None):
        directory = self.run_dir / "models" / loading_identity(model)
        directory.mkdir(parents=True, exist_ok=True)
        ready = directory / "ready.json"
        ready.unlink(missing_ok=True)
        self.alias = "lexverse-" + loading_identity(model)
        self.api_key = "lexverse-local-" + secrets.token_hex(24)
        settings = {"model": asdict(model), "api_key": self.api_key, "alias": self.alias,
                    "ready_path": str(ready.resolve()), "request_timeout": self.request_timeout,
                    "previous_source": (previous or {}).get("source")}
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        # Transformers/vLLM load only the resolved snapshot, including in offline mode.
        if model.load_parameters.get("local_files_only"):
            env["HF_HUB_OFFLINE"] = "1"
        with (directory / "worker.log").open("ab") as log:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "lexverse.providers.local"], stdin=subprocess.PIPE,
                stdout=log, stderr=log, env=env, start_new_session=True,
            )
        try:
            self.process.stdin.write((json.dumps(settings) + "\n").encode())
            self.process.stdin.close()
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise ConfigError(f"local model loading failed; see {directory / 'worker.log'}")
                if ready.exists():
                    info = json.loads(ready.read_text())
                    with urlopen(f"http://127.0.0.1:{info['port']}/health", timeout=2) as response:
                        if response.status == 200:
                            self.worker_identity, self.worker_info = loading_identity(model), info
                            return
                time.sleep(0.1)
            raise ConfigError(f"model startup exceeded {self.startup_timeout}s; see {directory / 'worker.log'}")
        except BaseException:
            self.stop_worker()
            raise

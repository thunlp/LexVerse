"""Resolve local/HF weights and serve Transformers/vLLM inference in one worker."""
from __future__ import annotations

import hashlib
import json
import os
import select
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lexverse.config import ConfigError, ModelConfig, is_local_path, validate_local_generation
from lexverse.runtime.results import atomic_write_json


def resolve_source(model: ModelConfig, previous: dict | None = None) -> tuple[Path, dict]:
    options = model.load_parameters
    if is_local_path(model.name):
        path = Path(model.name)
        source = {"type": "local", "path": str(path)}
    else:
        from huggingface_hub import snapshot_download
        revision = (previous or {}).get("resolved_revision") or options.get("revision", "main")
        path = Path(snapshot_download(
            repo_id=model.name, revision=revision, cache_dir=options.get("cache_dir"),
            local_files_only=options.get("local_files_only", False),
        ))
        source = {"type": "hf", "repo_id": model.name,
                  "requested_revision": options.get("revision", "main"),
                  "resolved_revision": path.name, "path": str(path)}
    if not path.is_dir():
        raise ConfigError(f"model directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise ConfigError(f"complete model config.json missing: {path}")
    if not any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt", "qwen.tiktoken")):
        raise ConfigError(f"model tokenizer files missing: {path}")
    weights = list(path.glob("model*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if not weights:
        raise ConfigError(f"complete model weights missing (adapters/GGUF are unsupported): {path}")
    for index in path.glob("*.index.json"):
        mapping = json.loads(index.read_text()).get("weight_map", {})
        if any(not (path / shard).is_file() for shard in mapping.values()):
            raise ConfigError(f"model shard missing: {index}")
    # Deliberately fingerprints metadata/indexes, not the large weight contents.
    files = sorted({*path.glob("*.json"), *path.glob("*.jinja")})
    fingerprint = hashlib.sha256()
    for file in files:
        fingerprint.update(file.name.encode())
        fingerprint.update(file.read_bytes())
    source["metadata_fingerprint"] = fingerprint.hexdigest()
    if previous and any(source.get(key) != previous.get(key) for key in ("type", "resolved_revision", "metadata_fingerprint")):
        raise ConfigError("model source changed; start a new run")
    return path, source


class LocalBackend:
    def __init__(self, model: ModelConfig, path: Path):
        from transformers import AutoTokenizer
        self.config = model
        config_path = path / "config.json"
        model_type = json.loads(config_path.read_text()).get("model_type") if config_path.is_file() else None
        self.legacy_qwen = model_type == "qwen"
        trust = model.load_parameters.get("trust_remote_code", False)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=trust, local_files_only=True)
        if model.input.get("chat_template"):
            self.tokenizer.chat_template = Path(model.input["chat_template"]).read_text(encoding="utf-8")
        if model_type == "internlm" and not self.tokenizer.chat_template:
            # Match the original InternLM build_inputs, including exactly one BOS.
            self.tokenizer.chat_template = (
                "{{ bos_token }}{% for message in messages %}"
                "{{ {'system': '<|System|>:', 'user': '<|User|>:', 'assistant': '<|Bot|>:'}[message['role']] }}"
                "{{ message['content'] }}{% if message['role'] == 'assistant' %}<eoa>{% endif %}{{ '\\n' }}"
                "{% endfor %}{% if add_generation_prompt %}<|Bot|>:{% endif %}"
            )
        if model.input.get("format", "chat") == "chat" and not self.tokenizer.chat_template and not self.legacy_qwen:
            raise ConfigError("chat model has no chat template; supply input.chat_template or use input.format: text")
        loading = {k: v for k, v in model.load_parameters.items() if k not in {"revision", "cache_dir", "local_files_only"}}
        if model.provider == "transformers":
            import torch
            from transformers import AutoModelForCausalLM
            dtype = loading.pop("dtype", "auto")
            dtype = "auto" if dtype == "auto" else getattr(torch, dtype)
            device = loading.pop("device_map", "auto")
            if device == "auto":
                device = "auto" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            if device == "mps" and not torch.backends.mps.is_available():
                raise ConfigError("MPS requested but unavailable in this Python process")
            if device == "cuda" and not torch.cuda.is_available():
                raise ConfigError("CUDA requested but unavailable in this Python process")
            self.model = AutoModelForCausalLM.from_pretrained(
                str(path), torch_dtype=dtype, device_map=device, local_files_only=True, **loading,
            ).eval()
            self.context_limit = getattr(self.model.config, "max_position_embeddings", self.tokenizer.model_max_length)
            self.device = str(self.model.device)
            self.dtype = str(self.model.dtype)
        else:
            from vllm import LLM
            # Render chat ourselves exactly once; vLLM receives token IDs.
            self.model = LLM(model=str(path), tokenizer=str(path), generation_config="vllm", **loading)
            self.context_limit = self.model.llm_engine.model_config.max_model_len
            self.device = "vllm"
            self.dtype = str(self.model.llm_engine.model_config.dtype)

    def generate(self, messages: list[dict], options: dict) -> dict:
        validate_local_generation(options)
        ids = self.encode(messages)
        max_tokens = options.get("max_tokens", 512)
        if not ids or len(ids) + max_tokens > self.context_limit:
            raise ConfigError(f"input plus requested output exceeds model context ({self.context_limit} tokens)")
        temperature = options.get("temperature", 0)
        if self.config.provider == "transformers":
            import torch
            kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0,
                      "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id}
            if temperature > 0:
                kwargs.update(temperature=temperature, top_p=options.get("top_p", 1))
            if self.legacy_qwen:
                kwargs["eos_token_id"] = [self.tokenizer.im_end_id, self.tokenizer.im_start_id, self.tokenizer.eod_id]
                if kwargs["pad_token_id"] is None:
                    kwargs["pad_token_id"] = self.tokenizer.eod_id
            inputs = torch.tensor([ids], device=self.model.device)
            with torch.inference_mode():
                result = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs), **kwargs)
            tokens = result[0, len(ids):].tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            stopped = self.legacy_qwen and tokens and tokens[-1] in kwargs["eos_token_id"]
            finish = "length" if len(tokens) >= max_tokens and not stopped else "stop"
        else:
            from vllm import SamplingParams
            params = SamplingParams(temperature=temperature, max_tokens=max_tokens, top_p=options.get("top_p", 1))
            result = self.model.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0].outputs[0]
            tokens, text, finish = result.token_ids, result.text, result.finish_reason
        return {"content": text, "finish_reason": finish,
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(tokens), "total_tokens": len(ids) + len(tokens)}}

    def encode(self, messages: list[dict]) -> list[int]:
        if not messages or any(not isinstance(m, dict) or not isinstance(m.get("content"), str)
                               or m.get("role") not in {"system", "user", "assistant"} for m in messages):
            raise ConfigError("local generation requires text system/user/assistant messages")
        if self.config.input.get("format", "chat") == "text":
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise ConfigError("input.format: text requires exactly one user message")
            return self.tokenizer.encode(messages[0]["content"])
        if self.legacy_qwen and not self.tokenizer.chat_template:
            # Match the original Qwen ChatML token construction, without truncation.
            if messages[0]["role"] != "system":
                messages = [{"role": "system", "content": "You are a helpful assistant."}, *messages]
            newline = self.tokenizer.encode("\n", allowed_special=set())
            ids = []
            for message in messages:
                if ids:
                    ids.extend(newline)
                ids.append(self.tokenizer.im_start_id)
                ids.extend(self.tokenizer.encode(message["role"], allowed_special=set()))
                ids.extend(newline)
                ids.extend(self.tokenizer.encode(message["content"], allowed_special=set()))
                ids.append(self.tokenizer.im_end_id)
            ids.extend(newline)
            ids.append(self.tokenizer.im_start_id)
            ids.extend(self.tokenizer.encode("assistant", allowed_special=set()))
            ids.extend(newline)
            return ids
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            **self.config.input.get("chat_template_kwargs", {}),
        )


def main() -> None:
    settings = json.loads(sys.stdin.readline())
    model = ModelConfig(**settings["model"])
    path, source = resolve_source(model, settings.get("previous_source"))
    backend = LocalBackend(model, path)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.respond(200, {"status": "ready"})
            else:
                self.respond(404, {"error": {"message": "unknown endpoint"}})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self.respond(404, {"error": {"message": "unknown endpoint"}})
            if self.headers.get("Authorization") != "Bearer " + settings["api_key"]:
                return self.respond(401, {"error": {"message": "invalid runtime credential"}})
            done = threading.Event()
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if payload.get("model") != settings["alias"] or payload.get("stream"):
                    raise ConfigError("unknown model or streaming requested")
                options = {k: v for k, v in payload.items() if k not in {"model", "messages", "stream", "response_format"}}
                response_format = payload.get("response_format")
                if response_format and response_format != {"type": "json_object"}:
                    raise ConfigError("only json_object response hints are accepted; schema constraints are unsupported")
                # json_object is a hint: native evaluator prompts/validators control JSON correctness.
                timeout = settings["request_timeout"]
                deadline = time.monotonic() + timeout if timeout is not None else None
                def watch_request():
                    while not done.wait(0.2):
                        if deadline is not None and time.monotonic() >= deadline:
                            os._exit(124)
                        readable, _, _ = select.select([self.connection], [], [], 0)
                        if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                            os._exit(125)  # Abandoned generation cannot contaminate the next request.
                threading.Thread(target=watch_request, daemon=True).start()
                with lock:
                    output = backend.generate(payload["messages"], options)
                if deadline is not None and time.monotonic() >= deadline:
                    os._exit(124)
                self.respond(200, {
                    "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion", "created": int(time.time()),
                    "model": settings["alias"], "usage": output["usage"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": output["content"]},
                                 "finish_reason": output["finish_reason"]}],
                })
            except (ConfigError, ValueError, KeyError) as exc:
                self.respond(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except Exception:
                # Preserve neither auth headers nor prompts in error messages.
                self.respond(500, {"error": {"message": "local inference failed; check model compatibility"}})
            finally:
                done.set()

        def respond(self, status, data):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # Never log headers, credentials or user messages.

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    import importlib.metadata
    versions = {}
    for package in ("torch", "transformers", "huggingface_hub", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    atomic_write_json(Path(settings["ready_path"]), {
        "port": server.server_port, "source": source, "versions": versions,
        "device": backend.device, "dtype": backend.dtype, "context_limit": backend.context_limit,
    })
    print(f"loaded provider={model.provider} device={backend.device} dtype={backend.dtype}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

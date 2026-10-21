"""Run native MILE Law_Fact generation in an isolated subprocess per Trial."""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

from lexverse.environments.base import ExecutionEnvironment
from lexverse.interaction.schema import EnvironmentResult
from lexverse.runtime.executor import run_subprocess
from lexverse.runtime.results import atomic_write_json
from . import UPSTREAM
from .tasks import PROFILE_KEYS, max_turns


def connection_spec(provider):
    return {
        "model": provider.model,
        "base_url": provider.base_url,
        "generation_parameters": dict(provider.default_config),
        "timeout_sec": provider.timeout_sec,
        "max_retries": 0,
    }


async def run_bridge(root, settings, connections, work_dir, timeout):
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        work_dir / "bridge_input.json",
        {**settings, "source_root": str(Path(root).resolve())},
    )
    env = dict(os.environ)
    env["LEXVERSE_MASER_CONNECTION_KEYS"] = json.dumps({
        key: value.api_key for key, value in connections.items()
    })
    package_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = os.pathsep.join([str(package_root), str(Path(root).resolve() / "src")])
    return await run_subprocess(
        [sys.executable, "-m", "lexverse.benchmarks.maser.environment", "bridge_input.json"],
        work_dir, timeout, env=env,
        expected_artifacts={
            "output": "session.json",
            "model_calls": "model_calls.json",
            "dialog_history": "dialog_history.jsonl",
            "native_all_info": "native_all_info.jsonl",
        },
    )


class NativeTransport:
    def __init__(self, settings, directory=None):
        self.connections = settings["connections"]
        self.keys = json.loads(os.environ.get("LEXVERSE_MASER_CONNECTION_KEYS", "{}"))
        self.calls = []
        self.lock = threading.Lock()
        self.directory = directory

    def record(self, value):
        with self.lock:
            self.calls.append(value)
            if self.directory is not None:
                atomic_write_json(self.directory / "model_calls.json", {"calls": self.calls})

    def request(self, model, messages, *, origin=None, native_model=None, **kwargs):
        from openai import OpenAI

        spec = self.connections[model]
        options = dict(spec.get("generation_parameters", {})) if model == "lawyer" else dict(kwargs)
        record = {
            "role": model,
            "model": model,
            "origin": origin,
            "native_model": native_model,
            "request_model": spec["model"],
            "messages": messages,
            "parameters": options,
        }
        timeout_kw = {"timeout": spec["timeout_sec"]} if spec.get("timeout_sec") is not None else {}
        try:
            with OpenAI(
                api_key=self.keys.get(model) or "unused",
                base_url=spec.get("base_url"),
                max_retries=0,
                **timeout_kw,
            ) as client:
                response = client.chat.completions.create(
                    model=spec["model"], messages=messages, **options
                )
                content = response.choices[0].message.content
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("model returned empty or invalid content")
        except Exception as exc:
            self.record({
                **record,
                "error_type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
            })
            raise RuntimeError(f"model request failed: {type(exc).__name__}") from None
        self.record({
            **record,
            "response": response.choices[0].message.content,
            "raw_response": response.model_dump(),
            "usage": response.usage.model_dump() if response.usage else {},
            "finish_reason": response.choices[0].finish_reason,
        })
        return response

    def call(self, model, messages, *, origin=None, **kwargs):
        return self.request(model, messages, origin=origin, **kwargs).choices[0].message.content


class MILEEnvironment(ExecutionEnvironment):
    name = "mile_native"

    def __init__(self, config):
        self.config = config
        self.connections = {}
        self.max_turns = max_turns(config.generation)
        self.root = UPSTREAM.ensure_patched(
            Path(__file__).with_name("patches"),
            offline=bool(config.generation.get("offline", False)),
        )

    async def run(self, task, work_dir):
        work_dir = Path(work_dir).resolve()
        settings = {
            "case_id": int(task.source.sample_id),
            "profile": task.raw_payload["profile"],
            "max_turns": self.max_turns,
            "connections": {key: connection_spec(value) for key, value in self.connections.items()},
        }
        await run_bridge(
            self.root, settings, self.connections, work_dir,
            self.config.generation.get("timeout_sec"),
        )
        session = json.loads((work_dir / "session.json").read_text(encoding="utf-8"))
        calls = json.loads((work_dir / "model_calls.json").read_text(encoding="utf-8"))["calls"]
        history = session["dialog_history"]
        artifact_files = {
            "session": "session.json",
            "model_calls": "model_calls.json",
            "dialog_history": "dialog_history.jsonl",
            "native_all_info": "native_all_info.jsonl",
            "stdout": "stdout.log",
            "stderr": "stderr.log",
        }
        return EnvironmentResult(
            status=session["status"],
            answer=history[-1]["content"] if session["status"] == "completed" else None,
            dialog_history=history,
            trace=calls,
            final_state={
                "native_status": session["status"],
                "termination_reason": session.get("termination_reason"),
                "error_type": session.get("error_type"),
            },
            artifacts={
                name: str(work_dir / filename)
                for name, filename in artifact_files.items()
                if (work_dir / filename).is_file()
            },
        )


def bootstrap(root):
    root = Path(root) / "src"
    for name in ("utils", "agents", "engine", "legal"):
        package = ModuleType(name)
        package.__path__ = [str(root / name)]
        sys.modules[name] = package
    loaded = {}
    module_names = (
        "utils.register", "engine.base_engine", "engine.gpt", "agents.base_agent",
        "agents.lawyer_fact", "agents.plaintiff_fact", "agents.supervisor",
        "legal.law_fact", "run_law_fact",
    )
    for name in module_names:
        spec = importlib.util.spec_from_file_location(name, root / f"{name.replace('.', '/')}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


def install_connections(loaded, settings, transport):
    registry = loaded["utils.register"].registry
    native_gpt = loaded["engine.gpt"].GPTEngine

    class QwenFactory:
        def __init__(self, api_key, model_name, seed):
            self.model_name = settings["connections"]["lawyer"]["model"]

        def get_response(self, messages):
            return transport.call("lawyer", messages, origin="lawyer")

    class GPTFactory(native_gpt):
        def __init__(self, *args, **kwargs):
            caller = inspect.currentframe().f_back.f_globals["__name__"].rsplit(".", 1)[-1]
            role = "client" if caller == "plaintiff_fact" else "supervisor"
            origin = {
                "plaintiff_fact": "client",
                "guider": "guide",
                "styler": "style",
                "tamperer": "temper",
            }.get(caller, caller)
            kwargs["openai_api_key"] = "unused"
            super().__init__(*args, **kwargs)
            self.client.close()

            def create(model, messages, **parameters):
                return transport.request(
                    role, messages, origin=origin, native_model=model, **parameters
                )

            completions = SimpleNamespace(create=create)
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    registry.register("Engine.Qwen", QwenFactory)
    registry.register("Engine.GPT", GPTFactory)


def role_options(spec):
    return {"model": spec["model"], **spec.get("generation_parameters", {})}


def run_generation(settings, directory, *, transport=None):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("dialog_history.jsonl", "native_all_info.jsonl", "model_calls.json"):
        (directory / filename).unlink(missing_ok=True)
    session = {
        "case_id": settings["case_id"],
        "status": "failed",
        "max_turns": settings["max_turns"],
        "dialog_history": [],
        "revise_history": [],
        "termination_reason": None,
        "error_type": None,
    }
    profile = {key: settings["profile"][key] for key in PROFILE_KEYS}
    staged_row = {**profile, "id": settings["case_id"], "reason": None, "laws": None}
    atomic_write_json(directory / "staged_input.json", [staged_row])

    root = Path(settings["source_root"]).resolve()
    loaded = bootstrap(root)
    transport = transport or NativeTransport(settings, directory=directory)
    transport.directory = directory
    install_connections(loaded, settings, transport)

    try:
        loaded["run_law_fact"].run_case(
            case_database=directory / "staged_input.json",
            save_path=directory / "dialog_history.jsonl",
            all_info_path=directory / "native_all_info.jsonl",
            max_turns=settings["max_turns"],
            plaintiff_config=role_options(settings["connections"]["client"]),
            supervisor_config=role_options(settings["connections"]["supervisor"]),
        )
        lines = (directory / "native_all_info.jsonl").read_text(encoding="utf-8").splitlines()
        native_all = json.loads(lines[0])
        history = native_all["dialog_history"]
        if len(history) < 3 or history[-1]["role"] != "Lawyer" or not history[-1]["content"].strip():
            raise ValueError("MILE native final draft missing")
        termination_reason = "end_marker" if "<询问结束>" in history[-2]["content"] else "max_turns"
        session.update(
            status="completed",
            dialog_history=history,
            revise_history=native_all["revise_history"],
            termination_reason=termination_reason,
        )
    except BaseException as exc:
        session.update(status="failed", error_type=type(exc).__name__)
    finally:
        atomic_write_json(directory / "model_calls.json", {"calls": transport.calls})
        atomic_write_json(directory / "session.json", session)
    return session


def main():
    settings = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    session = run_generation(settings, Path.cwd())
    if session["status"] != "completed":
        print(f"native generation failed: {session['error_type']}", file=sys.stderr)
        raise SystemExit(1)


MASEREnvironment = MILEEnvironment


if __name__ == "__main__":
    main()

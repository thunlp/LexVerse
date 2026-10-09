from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

from lexverse.environments.base import ExecutionEnvironment
from lexverse.interaction.schema import EnvironmentResult
from lexverse.providers.openai_compatible import OpenAICompatibleProvider
from lexverse.runtime.executor import run_subprocess
from lexverse.runtime.errors import TrialError
from lexverse.runtime.results import atomic_write_json
from . import UPSTREAM


def connection_spec(provider):
    return {"model": provider.model, "base_url": provider.base_url,
            "generation_parameters": provider.default_config,
            "timeout_sec": provider.timeout_sec, "max_retries": provider.max_retries,
            "local": hasattr(provider, "runtime")}


def load_metrics():
    from dlawbench.paths import PROJECT_ROOT
    spec = importlib.util.spec_from_file_location("dlawbench_public_metrics", PROJECT_ROOT / "scripts/metrics.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_response(text, messages, case):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("judge output must be a JSON object")
    if "memo_expectation_scores" in messages[-1]["content"]:
        for key, reference in (("memo_expectation_scores", "memo_expectations"),
                               ("expected_action_scores", "expected_actions")):
            scores = value.get(key)
            if (not isinstance(scores, list) or len(scores) != len(case["evaluation"][reference])
                    or any(type(score) not in {int, float} or score not in (0, 1) for score in scores)):
                raise ValueError(f"missing or invalid {key}")
    else:
        for section, keys in {"discovery": ["covered", "missed"],
                              "accuracy": ["correct", "incorrect"],
                              "grounding": ["grounded_claims", "ungrounded_claims"]}.items():
            if not isinstance(value.get(section), dict) or any(
                not isinstance(value[section].get(key), list) for key in keys
            ):
                raise ValueError(f"missing or invalid {section}")
        count = value["grounding"].get("total_claims")
        if (type(count) is not int or count < 0
                or count != len(value["grounding"]["grounded_claims"]) + len(value["grounding"]["ungrounded_claims"])):
            raise ValueError("invalid total_claims")
        ids = {fact["id"] for fact in case["facts"]}
        covered, missed = value["discovery"]["covered"], value["discovery"]["missed"]
        if (any(not isinstance(fid, str) for fid in covered + missed)
                or set(covered) & set(missed) or set(covered + missed) != ids
                or len(covered + missed) != len(ids)):
            raise ValueError("DAG judge must cover each reference fact exactly once")
        correct = value["accuracy"]["correct"]
        incorrect = [item.get("fact_id") for item in value["accuracy"]["incorrect"] if isinstance(item, dict)]
        if set(correct + incorrect) != set(covered) or len(correct + incorrect) != len(covered):
            raise ValueError("DAG accuracy coverage incomplete")


async def run_bridge(root, mode, settings, connections, work_dir, timeout):
    work_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(work_dir / "bridge_input.json", settings)
    env = dict(os.environ)
    env["LEXVERSE_DLAW_CONNECTION_KEYS"] = json.dumps({key: value.api_key for key, value in connections.items()})
    # Include the installed package root explicitly; never rely on a developer cwd.
    package_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = os.pathsep.join([str(package_root), str(root.resolve() / "src")])
    return await run_subprocess(
        [sys.executable, "-m", "lexverse.benchmarks.dlawbench.environment", mode, "bridge_input.json"],
        work_dir, timeout, env=env,
        expected_artifacts={"output": "session.json" if mode == "generate" else "grading.json"},
    )


class NativeTransport:
    def __init__(self, settings):
        self.connections = settings["connections"]
        self.keys = json.loads(os.environ["LEXVERSE_DLAW_CONNECTION_KEYS"])
        self.calls = []
        self.lock = threading.Lock()
        self.case = settings["case"]
        self.evaluation = "session" in settings

    def call(self, model, messages, **kwargs):
        spec = self.connections[model]
        # Explicit ModelConfig options override the native call defaults.
        from dlawbench import api_utils
        options = {"max_tokens": api_utils.default_max_tokens_for_model(spec["model"]),
                   **kwargs, **spec["generation_parameters"]}
        if "reasoning_effort" not in spec["generation_parameters"] and not api_utils.use_responses_api(spec["model"]):
            options.pop("reasoning_effort", None)
        native_timeout = options.pop("timeout", None)
        if spec["local"]:
            options = {key: value for key, value in options.items() if key in {"temperature", "top_p", "max_tokens"}}
        provider = OpenAICompatibleProvider(
            name=model, model=spec["model"], api_key=self.keys[model], base_url=spec["base_url"],
            timeout_sec=(spec["timeout_sec"] if spec["timeout_sec"] is not None else
                         native_timeout if native_timeout is not None else api_utils.DEFAULT_TIMEOUT),
            max_retries=spec["max_retries"],
        )

        async def request():
            try:
                if not spec["local"] and api_utils._is_gpt5_family(spec["model"]):
                    params = {"model": spec["model"], **{
                        key: value for key, value in options.items()
                        if key in {"temperature", "max_tokens", "top_p", "seed", "stop"}
                    }}
                    extra_body = dict(api_utils.MODEL_PARAMS.get(spec["model"], {}))
                    extra_body.update({key: value for key, value in options.items()
                                       if key not in api_utils._STANDARD_KWARGS and value is not None})
                    args = (str(provider._get_client().base_url), self.keys[model], params,
                            messages, provider.timeout_sec, provider.max_retries)
                    if api_utils.use_responses_api(spec["model"]):
                        try:
                            result = await asyncio.to_thread(api_utils.call_responses_api, *args,
                                                             extra_body=extra_body)
                            content = api_utils.extract_content_from_responses(result)
                            if not content or not content.strip():
                                raise ValueError("Responses API returned empty content")
                            return content
                        except Exception:
                            # Match upstream fallback for request, decoding, and empty-text failures.
                            pass
                    result = await asyncio.to_thread(api_utils.call_chat_completion, *args,
                                                     extra_body=extra_body)
                    return api_utils.extract_content(result)
                return (await provider.generate(messages, config=options)).content
            finally:
                if provider._client is not None:
                    await provider._client.close()

        try:
            response = asyncio.run(request())
        except Exception as exc:
            # Upstream writes exceptions to stderr; expose no connection secrets.
            raise RuntimeError(f"model request failed: {type(exc).__name__}") from None
        with self.lock:
            self.calls.append({"model": model, "messages": messages, "parameters": options,
                               "response": response})
        if self.evaluation:
            validate_response(response, messages, self.case)
        return {}, response


class DLawBenchEnvironment(ExecutionEnvironment):
    name = "dlawbench_native"

    def __init__(self, config):
        self.config = config
        self.connections = {}
        self.root = UPSTREAM.ensure(offline=bool(config.generation.get("offline", False)))

    async def run(self, task, work_dir):
        work_dir = work_dir.resolve()
        settings = {"case": task.raw_payload, "metadata": task.metadata,
                    "max_turns": self.config.generation.get("max_turns", 10),
                    "lawyer_name": self.config.models.roles.get("lawyer", self.config.models.default).name,
                    "client_name": self.config.models.roles.get("client", self.config.models.default).name,
                    "connections": {key: connection_spec(value) for key, value in self.connections.items()}}
        await run_bridge(self.root, "generate", settings, self.connections, work_dir,
                         self.config.generation.get("timeout_sec"))
        session = json.loads((work_dir / "session.json").read_text())
        if session["status"] == "api_error":
            raise TrialError("DLawBench api_error; see session.json")
        return EnvironmentResult(
            status="completed", answer=session.get("memo"), dialog_history=session["log"],
            final_state={"native_status": session["status"], "has_memo": bool(session.get("memo"))},
            artifacts={name: str(work_dir / filename) for name, filename in {
                "session": "session.json", "memo": "memo.txt", "dialog_history": "dialog_history.jsonl",
                "native_status": "status.json", "model_calls": "model_calls.json", "stdout": "stdout.log", "stderr": "stderr.log",
            }.items()},
        )


def main():
    from dlawbench import agent_environment, client_npc, evaluate, panel_evaluate
    from dlawbench.case_utils import normalize_case

    mode, filename = sys.argv[1:]
    settings = json.loads(Path(filename).read_text())
    settings["case"] = normalize_case(settings["case"])
    case = settings["case"]
    transport = NativeTransport(settings)
    for module in (agent_environment, client_npc, evaluate):
        module.call_model = transport.call
    if mode == "generate":
        persona = client_npc.load_persona(settings["metadata"]["persona_key"])
        environment = agent_environment.AgentEnvironment(
            case, persona, "lawyer", "client", max_turns=settings["max_turns"],
        )
        started = time.monotonic()
        try:
            session = environment.run()
        except Exception as exc:
            session = {"case_id": case["case_id"], "persona": persona["name_cn"],
                       "status": "api_error", "memo": environment.memo_content, "log": environment.log,
                       "turns_used": environment.turn, "max_turns": settings["max_turns"],
                       "error_type": type(exc).__name__,
                       "elapsed_seconds": round(time.monotonic() - started, 1),
                       "timestamp": datetime.now(timezone.utc).isoformat()}
        session.update(lawyer_model=settings["lawyer_name"],
                       client_model=settings["client_name"],
                       **{key: settings["metadata"][key] for key in ("case_file", "case_line", "jurisdiction")})
        atomic_write_json(Path("session.json"), session)
        Path("memo.txt").write_text(session.get("memo") or "", encoding="utf-8")
        Path("dialog_history.jsonl").write_text(
            "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in session["log"]), encoding="utf-8",
        )
        atomic_write_json(Path("status.json"), {"native_status": session["status"],
                                               "has_memo": bool(session.get("memo"))})
    else:
        session = settings["session"]
        result = None
        error = None
        try:
            if settings["mode"] == "panel":
                panel = settings["panel"]
                atomic_write_json(Path("panel.json"), {"infrastructure": {"judge_panel": panel}})
                active, recused = panel_evaluate.resolve_judge_panel(session["lawyer_model"], "panel.json")
                if not active:
                    raise ValueError("native recusal left no active judge")
                result = panel_evaluate.evaluate_session_panel(session, case, "panel.json")
                failed = [judge for judge in active if "error" in result["per_judge"][judge]]
                if failed:
                    error = "judge coverage incomplete: " + ", ".join(failed)
            else:
                result = evaluate.evaluate_session(session, case, "judge")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        metrics = None
        if result is not None and error is None:
            native_metrics = load_metrics()
            row = native_metrics.session_metrics(Path("session.json"), dict(session, evaluation=result), case["jurisdiction"])
            metrics = {key: row[key] for key in native_metrics.METRIC_KEYS}
        atomic_write_json(Path("grading.json"), {"native_result": result, "metrics": metrics, "error": error})
    atomic_write_json(Path("model_calls.json"), {"calls": transport.calls})


if __name__ == "__main__":
    main()

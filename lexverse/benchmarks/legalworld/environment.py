"""Run the upstream pipeline in one isolated process per Trial."""
from __future__ import annotations

from contextvars import ContextVar
import json
import os
from pathlib import Path
import sys
import time

from lexverse.environments.base import ExecutionEnvironment
from lexverse.interaction.schema import EnvironmentResult
from lexverse.runtime.executor import run_subprocess
from lexverse.runtime.results import atomic_write_json
from . import UPSTREAM
from .tasks import ensure_dataset, stage_range
from .resources import law_resources, case_resources, exclude_benchmark_cases


ACTIVE_ROLE = ContextVar("legalworld_model_role", default="simulation")


def connection_spec(provider):
    return {"model": provider.model, "base_url": provider.base_url,
            "generation_parameters": provider.default_config,
            "timeout_sec": provider.timeout_sec, "max_retries": provider.max_retries}


def protect_source(root):
    root = root.resolve()

    def audit(event, args):
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            flags = args[2] or 0
            if path.is_relative_to(root) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise PermissionError("Legal-world source cache is read-only during a Trial")
        if event in {"os.mkdir", "os.remove", "os.rmdir", "os.rename"}:
            for value in args[:2] if event == "os.rename" else args[:1]:
                if isinstance(value, (str, bytes, os.PathLike)) and Path(os.fsdecode(value)).resolve().is_relative_to(root):
                    raise PermissionError("Legal-world source cache is read-only during a Trial")

    sys.addaudithook(audit)


def install_connections(settings, mode):
    from camel.models import ModelFactory
    from src.agents.base_agent import BaseAgent
    connections = settings["connections"]
    keys = json.loads(os.environ.pop("LEXVERSE_LEGALWORLD_KEYS", "{}"))
    native_create = ModelFactory.create
    native_activate = BaseAgent.activate
    request_errors = []

    def activate(agent, *args, **kwargs):
        token = ACTIVE_ROLE.set("lawyer" if agent.agent_id == "lawyer" else "simulation")
        try:
            return native_activate(agent, *args, **kwargs)
        finally:
            ACTIVE_ROLE.reset(token)

    def create(*args, **kwargs):
        role = "judge" if mode == "evaluate" else ACTIVE_ROLE.get()
        connection = connections[role]
        parameters = dict(kwargs.get("model_config_dict") or {})
        parameters.update(connection["generation_parameters"])
        kwargs.update(model_type=connection["model"], url=connection["base_url"], api_key=keys[role],
                      model_config_dict=parameters,
                      max_retries=connection["max_retries"])
        if connection["timeout_sec"] is not None:
            kwargs["timeout"] = connection["timeout_sec"]
        backend = native_create(*args, **kwargs)
        native_run = backend.run

        def run(*run_args, **run_kwargs):
            try:
                return native_run(*run_args, **run_kwargs)
            except Exception as exc:
                request_errors.append({"role": role, "error_type": type(exc).__name__})
                atomic_write_json(Path("model_errors.json"), request_errors)
                raise RuntimeError(f"Legal-world {role} request failed: {type(exc).__name__}") from None

        backend.run = run
        return backend

    BaseAgent.activate = activate
    ModelFactory.create = staticmethod(create)
    return request_errors


async def run_native_process(mode, settings, connections, work_dir, timeout):
    work_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(work_dir / "bridge_input.json", settings)
    env = dict(os.environ)
    # Prevent ambient private skills, indexes, credentials and experiment flags from changing a run.
    for key in list(env):
        if key.startswith(("SIMLAW_", "LAW_", "OPENAI_")):
            env.pop(key)
    env["LEXVERSE_LEGALWORLD_KEYS"] = json.dumps({key: value.api_key for key, value in connections.items()})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    env["SIMLAW_ENABLE_LAW_RETRIEVAL"] = "false"
    law = settings.get("law_retrieval", {}) if mode == "generate" else {}
    if law:
        from lexverse.config import load_profile
        profile = load_profile(law["embedding_profile"])
        env.update(SIMLAW_ENABLE_LAW_RETRIEVAL="true",
                   LAW_RETRIEVAL_INDEX_DIR=law["index_dir"],
                   LAW_EMBEDDING_MODEL=law["embedding_model"],
                   LAW_EMBEDDING_API_KEY=profile.api_key,
                   LAW_EMBEDDING_API_BASE_URL=profile.base_url)

    return await run_subprocess(
        [sys.executable, "-m", "lexverse.benchmarks.legalworld.environment", mode, "bridge_input.json"],
        work_dir, timeout, env=env,
        expected_artifacts={"output": "pipeline_result.json" if mode == "generate" else "eval_result.json"},
    )


class LegalWorldEnvironment(ExecutionEnvironment):
    name = "legalworld_native"

    def __init__(self, config):
        self.config = config
        self.connections = {}
        self.law_settings = law_resources(config)[0]
        self.case_settings = case_resources(config)[0]
        self.root = UPSTREAM.ensure_patched(Path(__file__).with_name("patches"),
                                           offline=bool(config.generation.get("offline", False)))

    async def run(self, task, work_dir):
        work_dir = work_dir.resolve()
        native_dir = work_dir / "native"
        if native_dir.exists():
            native_dir.rename(work_dir / f"native-previous-{time.time_ns()}")
        settings = {**task.metadata, "law_retrieval": self.law_settings, "case_retrieval": self.case_settings, "dataset": str(ensure_dataset(offline=True)),
                    "root": str(self.root.resolve()),
                    "connections": {key: connection_spec(value) for key, value in self.connections.items()}}
        await run_native_process("generate", settings, self.connections, native_dir,
                         self.config.generation.get("timeout_sec"))
        result = json.loads((native_dir / "pipeline_result.json").read_text())
        stages = result["stages_completed"]
        expected = stage_range(task.metadata["start_stage"], task.metadata["end_stage"])
        if stages != expected or result.get("party_role") != task.metadata["party_role"]:
            raise ValueError("native pipeline identity or stage coverage mismatch")
        return EnvironmentResult(
            status="completed", answer=None,
            final_state={"party_role": result["party_role"], "stages_completed": stages},
            artifacts={"pipeline_result": str(native_dir / "pipeline_result.json"),
                       **{f"native:{path.relative_to(native_dir)}": str(path)
                          for path in sorted(native_dir.rglob("*")) if path.is_file()}},
        )


def monitor_retrieval(cls, method, errors, role):
    native = getattr(cls, method)

    def call(self, *args, **kwargs):
        try:
            return native(self, *args, **kwargs)
        except Exception as exc:
            errors.append({"role": role, "error_type": type(exc).__name__})
            atomic_write_json(Path("model_errors.json"), errors)
            raise RuntimeError(f"Legal-world {role} failed: {type(exc).__name__}") from None

    setattr(cls, method, call)


def install_case_retrieval(settings, request_errors):
    case = settings.get("case_retrieval", {})
    if not case:
        return
    docs = [json.loads(line) for line in Path(case["docs_path"]).read_text().splitlines() if line.strip()]
    kept, excluded = exclude_benchmark_cases(docs, json.loads(Path(settings["dataset"]).read_text()))
    atomic_write_json(Path("case_retrieval_audit.json"), {"source": case, "excluded_rows": excluded,
                      "remaining": len(kept), "policy": "all benchmark docket matches or >=80% long-text shingle containment"})
    if not kept:
        raise ValueError("case corpus empty after benchmark exclusion")
    local = Path("case_retrieval_docs.jsonl").resolve()
    local.write_text("".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in kept))
    from src.pipeline import stage_tool_registry, stage_tool_resolver
    from src.tools.legal.case_retrieval_tool import create_case_retrieval_tool, LocalCaseRetrievalEngine
    monitor_retrieval(LocalCaseRetrievalEngine, "search", request_errors, "case_retrieval")
    stage_tool_registry.REGISTERED_STAGE_TOOL_FACTORIES["search_cases"] = lambda agent: create_case_retrieval_tool(storage_path=str(local))
    manifest = stage_tool_resolver.load_stage_tool_manifest()
    manifest["tool_registry_refs"].append("search_cases")
    manifest["agent_type_defaults"]["lawyer"].append("search_cases")
    atomic_write_json(Path("effective_stage_tool_manifest.json"), manifest)


def main():
    mode, path = sys.argv[1:]
    settings = json.loads(Path(path).read_text())
    root = Path(settings["root"])
    protect_source(root)
    sys.path.insert(0, str(root / "backend"))
    sys.path.insert(0, str(root / "backend/src"))
    from src.data.data_loader import DataLoader
    loader = DataLoader(settings["dataset"])
    if not loader.get_case(settings["original_id"]):
        raise ValueError("original_id not present in pinned dataset")
    request_errors = install_connections(settings, mode)
    if mode == "generate":
        if settings.get("law_retrieval"):
            from src.tools.common.law_retrieval_tool import LawRetrievalTool
            monitor_retrieval(LawRetrievalTool, "_embed_query", request_errors, "embedding")
        install_case_retrieval(settings, request_errors)
        from src.pipeline.pipeline import LegalPipeline
        pipeline = LegalPipeline(
            loader, case_index=settings["original_id"], output_dir=str(Path.cwd()),
            start_stage=settings["start_stage"], end_stage=settings["end_stage"],
            party_role=settings["party_role"], enable_console_output=False,
            evaluated_lawyer_model_name=settings["connections"]["lawyer"]["model"],
            fixed_simulation_model_name=settings["connections"]["simulation"]["model"],
        )
        try:
            pipeline.run()
            if request_errors:
                raise RuntimeError("native model requests failed; see model_errors.json")
        except Exception as exc:
            partial = pipeline._build_final_result()
            partial.update(status="execution_failed", error_type=type(exc).__name__)
            atomic_write_json(Path("pipeline_result.json"), partial)
            raise
        finally:
            pipeline.export_agent_prompts(str(Path("agent_prompts.json").resolve()))
    else:
        from src.eval.eval_pipeline import EvalPipeline
        evaluator = EvalPipeline(settings["pipeline_result"], loader, case_index=settings["original_id"],
                                 output_path=str(Path("eval_result.json").resolve()),
                                 judge_model_type=settings["connections"]["judge"]["model"])
        try:
            evaluator.run()
            if request_errors:
                raise RuntimeError("native judge requests failed; see model_errors.json")
        except Exception as exc:
            atomic_write_json(Path("eval_result.json"), {
                "case_id": settings["original_id"], "status": "scoring_failed",
                "error_type": type(exc).__name__, "stage_eval_results": evaluator.eval_results,
            })
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Native exceptions may contain credentials supplied by a provider.
        detail = f" (missing module: {exc.name})" if isinstance(exc, ModuleNotFoundError) else ""
        print(f"Legal-world environment failed: {type(exc).__name__}{detail}", file=sys.stderr)
        sys.exit(1)

from contextlib import AsyncExitStack, asynccontextmanager
import asyncio
import json
from pathlib import Path

from filelock import FileLock

from lexverse.agents.spec import AgentExecutionSpec
from lexverse.capabilities.base import execution_conditions, identity
from lexverse.capabilities.knowledge.index import KnowledgeIndex, build_index, preparation_report, verify_index
from lexverse.capabilities.mcp_client import open_knowledge_tools, open_mcp_tools
from lexverse.capabilities.registry import ResourceRegistry, load_registry
from lexverse.capabilities.skills import prepare_skills, tree_hashes
from lexverse.config import ConfigError
from lexverse.runtime.identity import execution_identity
from lexverse.runtime.paths import CAPABILITIES_ROOT, RunPaths
from lexverse.runtime.results import atomic_write_json


async def _embeddings(spec, cache, recorder):
    if spec.selection.knowledge and spec.retrieval.mode != "keyword":
        from lexverse.capabilities.knowledge.embeddings import create_embeddings
        return await asyncio.to_thread(create_embeddings, spec.retrieval.embedding, cache, recorder=recorder)
    return None


def validate_agent_resources(spec, resources, *, manifest=None):
    """Check saved resource identities and files without loading runtime models."""
    paths = RunPaths(spec.run_dir)
    if manifest is None:
        manifest = json.loads((paths.internal / "manifest.json").read_text())
    if identity(resources) != manifest.get("resource_hash"):
        raise ConfigError("Resource identity changed; start a new run")
    skills = resources["skills"]
    if skills["files"] != tree_hashes(spec.skills_root or paths.agent_fs / "skills"):
        raise ConfigError("Skill projection changed since the run started")
    for record in resources["knowledge"]:
        index = verify_index(Path(record["path"]))
        if index["index_hash"] != record["index_hash"]:
            raise ConfigError("Resource identity changed (knowledge); start a new run")


@asynccontextmanager
async def load_agent_resources(spec, resources, *, cache=None, adapter_factory=None, embedding_recorder=None):
    """Open the built-in agent's runtime connections and check live MCP schemas."""
    cache = cache or CAPABILITIES_ROOT
    registry = ResourceRegistry.model_validate(resources["registry"])
    async with AsyncExitStack() as stack:
        knowledge = await stack.enter_async_context(open_knowledge_tools(
            spec, resources, cache=cache, embedding_recorder=embedding_recorder,
        ))
        remote, discovery = await stack.enter_async_context(open_mcp_tools(
            registry, spec.selection.mcp, adapter_factory=adapter_factory,
        ))
        if identity(discovery) != identity(resources["mcp"]):
            raise ConfigError("Resource identity changed (mcp); start a new run")
        yield knowledge + remote


async def prepare_agent_resources(spec: AgentExecutionSpec, *, agent_name="lexverse", cache=None,
                                  adapter_factory=None) -> dict:
    """Prepare resource files once for a user task or benchmark template."""
    cache = cache or CAPABILITIES_ROOT
    run = spec.run_dir.resolve()
    paths = RunPaths(run)
    manifest_path = paths.internal / "manifest.json"
    ledger = None
    if agent_name == "lexverse":
        from lexverse.capabilities.middleware import RunLedger
        ledger = RunLedger(run, spec.limits)

    # Prevent concurrent preparation of the same run.
    with FileLock(str(paths.internal / "execution.lock"), timeout=0):
        try:
            # Validate the saved execution conditions.
            manifest = json.loads(manifest_path.read_text())
            execution = execution_identity(spec, agent_name=agent_name)
            if manifest.get("execution") and execution_conditions(manifest["execution"]) != execution_conditions(execution):
                raise ConfigError("Execution configuration, dependencies or inputs changed; start a new run")

            # Reuse prepared resources after validating files and MCP schemas.
            if manifest.get("resources"):
                resources = manifest["resources"]
                validate_agent_resources(spec, resources, manifest=manifest)
                if agent_name == "lexverse":
                    registry = ResourceRegistry.model_validate(resources["registry"])
                    async with open_mcp_tools(registry, spec.selection.mcp, adapter_factory=adapter_factory) as (_, discovery):
                        if identity(discovery) != identity(resources["mcp"]):
                            raise ConfigError("Resource identity changed (mcp); start a new run")
                return resources

            # Load the registry and prepare selected Skills.
            registry = load_registry(spec.selection.registry)
            skills = prepare_skills(registry, spec.selection.skills, cache, paths.agent_fs)

            # Prepare embeddings and bind input sources to the run snapshot.
            embedding = await _embeddings(spec, cache, ledger.record_embedding if ledger else None)
            input_source = Path(manifest["task"]["inputs"]).resolve() if "task" in manifest else None
            sources = [paths.inputs if source.resolve() == input_source else source for source in spec.selection.knowledge]
            async with AsyncExitStack() as stack:
                # Build or reuse knowledge indexes and record their identities.
                indexes, knowledge = [], []
                for source in sources:
                    scan = {}
                    path = await asyncio.to_thread(build_index, source, cache, spec.retrieval, embedding, scan_report=scan)
                    index = KnowledgeIndex(path, embedding, config=spec.retrieval)
                    stack.callback(index.close)
                    indexes.append(index)
                    knowledge.append({
                        "path": str(path), "index_hash": index.index_hash,
                        "retrieval": spec.retrieval.model_dump(mode="json"),
                        "preparation": await asyncio.to_thread(preparation_report, index, cache, skipped=scan["skipped"]),
                    })

                # Discover external MCP tools for the built-in agent.
                discovery = []
                if agent_name == "lexverse":
                    _, discovery = await stack.enter_async_context(open_mcp_tools(
                        registry, spec.selection.mcp, adapter_factory=adapter_factory,
                    ))

                # Prepare keyword search when enabled by the retrieval mode.
                keyword_cache = None
                if indexes and spec.retrieval.mode != "vector":
                    from lexverse.capabilities.knowledge.keyword import KeywordSearch
                    keyword_search = KeywordSearch(indexes, cache / "cache/keyword_search")
                    await asyncio.to_thread(keyword_search.prepare)
                    keyword_cache = str(keyword_search.path.resolve())

            # Freeze resource identities and persist the preparation manifest.
            skills["files"] = tree_hashes(spec.skills_root or paths.agent_fs / "skills")
            resources = {"registry": registry.model_dump(mode="json"), "skills": skills,
                         "knowledge": knowledge, "mcp": discovery, "keyword_cache": keyword_cache}
            manifest["resources"] = resources
            manifest["resource_hash"] = identity(resources)
            manifest["execution"] = execution
            atomic_write_json(manifest_path, manifest)
            return resources
        except (Exception, asyncio.CancelledError) as exc:
            result = json.loads((run / "result.json").read_text())
            cancelled = isinstance(exc, asyncio.CancelledError)
            result.update(status="cancelled" if cancelled else "failed",
                          stop_reason="cancelled" if cancelled else "preparation_failed", error_type=type(exc).__name__,
                          error_message=str(exc), run_dir=str(run), answer=None, artifacts=[], question=None,
                          trace_ref=str(paths.internal / "events.jsonl"))
            atomic_write_json(run / "result.json", result)
            if ledger:
                ledger.event("run_failed", error_type=type(exc).__name__)
            raise
        finally:
            if ledger:
                ledger.finish()

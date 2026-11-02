from contextlib import AsyncExitStack, asynccontextmanager
import asyncio
import json
import tempfile
from pathlib import Path

from filelock import FileLock

from lexverse.agents.base import AgentExecutionSpec
from lexverse.capabilities.base import execution_conditions, identity
from lexverse.capabilities.knowledge.index import KnowledgeIndex, build_index, preparation_report, verify_index
from lexverse.capabilities.mcp_client import open_knowledge_tools, open_mcp_tools
from lexverse.capabilities.skills import prepare_skills, tree_hashes
from lexverse.config import ConfigError
from lexverse.agents.base import execution_identity
from lexverse.utils.paths import CAPABILITIES_ROOT
from lexverse.models.trial.paths import RunPaths
from lexverse.utils.files import atomic_write_json


async def _embeddings(spec, cache, recorder):
    if spec.selection.knowledge and spec.retrieval.mode != "keyword":
        from lexverse.capabilities.knowledge.embeddings import create_embeddings
        return await asyncio.to_thread(create_embeddings, spec.retrieval.embedding, cache, recorder=recorder)
    return None


def prepared_resources(manifest):
    if manifest.get('resources_ref'):
        shared = json.loads(Path(manifest['resources_ref']).read_text())
        resources = shared['resources']
        if identity(resources) != manifest['resource_hash']:
            raise ConfigError('Shared resources changed; start a new run')
        return resources
    return manifest.get('resources')


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
async def load_agent_resources(spec, resources, *, cache=None, adapter_factory=None, embedding_recorder=None, environment=None):
    """Open the built-in agent's runtime connections and check live MCP schemas."""
    cache = cache or CAPABILITIES_ROOT
    async with AsyncExitStack() as stack:
        knowledge = await stack.enter_async_context(open_knowledge_tools(
            spec, resources, cache=cache, embedding_recorder=embedding_recorder,
        ))
        servers = [await environment.mcp_for_host(server) for server in spec.selection.mcp_servers] if environment else spec.selection.mcp_servers
        remote, discovery = await stack.enter_async_context(open_mcp_tools(servers, adapter_factory=adapter_factory))
        original = {server.name: server for server in spec.selection.mcp_servers}
        for record in discovery:
            if "url" in record:
                record["url"] = original[record["alias"]].url
        if resources["mcp"] is not None and identity(discovery) != identity(resources["mcp"]):
            raise ConfigError("Resource identity changed (mcp); start a new run")
        if resources["mcp"] is None:
            resources["mcp"] = discovery
            manifest_path = RunPaths(spec.run_dir).internal / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update(resources=resources, resource_hash=identity(resources))
            atomic_write_json(manifest_path, manifest)
        yield knowledge + remote


async def prepare_agent_resources(spec: AgentExecutionSpec, *, agent_name="vanilla", cache=None,
                                  adapter_factory=None) -> dict:
    """Prepare resource files once for a user task or benchmark template."""
    cache = cache or CAPABILITIES_ROOT
    run = spec.run_dir.resolve()
    paths = RunPaths(run)
    manifest_path = paths.internal / "manifest.json"
    ledger = None
    if agent_name == "vanilla":
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
                return resources

            # Project the explicitly selected Skill sources.
            skills = prepare_skills(spec.selection.skills, cache, paths.agent_fs)

            # Prepare embeddings and bind input sources to the run snapshot.
            embedding = await _embeddings(spec, cache, ledger.record_embedding if ledger else None)
            sources = spec.selection.knowledge
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

                # Prepare keyword search when enabled by the retrieval mode.
                keyword_cache = None
                if indexes and spec.retrieval.mode != "vector":
                    from lexverse.capabilities.knowledge.keyword import KeywordSearch
                    keyword_search = KeywordSearch(indexes, cache / "cache/keyword_search")
                    await asyncio.to_thread(keyword_search.prepare)
                    keyword_cache = str(keyword_search.path.resolve())

            # Freeze resource identities and persist the preparation manifest.
            skills["files"] = tree_hashes(spec.skills_root or paths.agent_fs / "skills")
            resources = {"mcp_servers": [server.model_dump(mode="json") for server in spec.selection.mcp_servers], "skills": skills,
                         "knowledge": knowledge, "mcp": None if spec.selection.mcp_servers else [],
                         "keyword_cache": keyword_cache}
            manifest["resources"] = resources
            manifest["resource_hash"] = identity(resources)
            manifest["execution"] = execution
            atomic_write_json(manifest_path, manifest)
            return resources
        except (Exception, asyncio.CancelledError) as exc:
            result = json.loads((RunPaths(run).worker_result).read_text())
            cancelled = isinstance(exc, asyncio.CancelledError)
            result.update(status="cancelled" if cancelled else "failed",
                          stop_reason="cancelled" if cancelled else "preparation_failed", error_type=type(exc).__name__,
                          error_message=str(exc), run_dir=str(run), answer=None, artifacts=[], question=None,
                          trace_ref=str(paths.internal / "events.jsonl"))
            atomic_write_json(RunPaths(run).worker_result, result)
            if ledger:
                ledger.event("run_failed", error_type=type(exc).__name__)
            raise
        finally:
            if ledger:
                ledger.finish()


def check_sources(run, selection, cache):
    from lexverse.capabilities.knowledge.loaders import scan_source
    from lexverse.capabilities.knowledge.index import is_index_directory, verify_index
    from lexverse.capabilities.skills import prepare_skills

    saved = json.loads((run / ".internal/manifest.json").read_text())["resources"]
    if [server.model_dump(mode="json") for server in selection.mcp_servers] != saved["mcp_servers"]:
        raise ConfigError("MCP configuration changed; start a new run")
    for source, record in zip(selection.knowledge, saved["knowledge"], strict=True):
        manifest = json.loads((Path(record["path"]) / "manifest.json").read_text())
        if is_index_directory(source):
            unchanged = verify_index(source)["index_hash"] == manifest["index_hash"]
        else:
            unchanged = scan_source(source)[0] == manifest["identity"]["files"]
        if not unchanged:
            raise ConfigError("Knowledge source changed; start a new run")
    with tempfile.TemporaryDirectory() as temporary:
        current = prepare_skills(selection.skills, cache, Path(temporary))
        if current["files"] != saved["skills"].get("execution_files", saved["skills"]["files"]):
            raise ConfigError("Skill source changed; start a new run")

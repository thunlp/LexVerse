from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import override

from lexverse.agents.atif_converter import (
    convert_langchain_messages_to_atif,
    write_atif_trajectory,
)
from lexverse.agents.base import BaseAgent
from lexverse.agents.factory import AgentFactory
from lexverse.agents.general_runner import build_mcp_server_configs
from lexverse.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    EnvVar,
    with_prompt_template,
)
from lexverse.agents.installed.claude_code import ClaudeCode
from lexverse.agents.installed.codex import Codex
from lexverse.agents.installed.gemini_cli import GeminiCli
from lexverse.agents.installed.opencode import OpenCode
from lexverse.agents.installed.openhands import OpenHands
from lexverse.agents.installed.qwen_code import QwenCode
from lexverse.capabilities.base import AgentExecutionSpec
from lexverse.commands.tasks import main as task_cli_main
from lexverse.config import ModelConfig
from lexverse.environments.base import BaseEnvironment
from lexverse.environments.factory import EnvironmentFactory
from lexverse.environments.local import LocalEnvironment
from lexverse.models.agent.context import AgentContext
from lexverse.models.agent.name import AgentName
from lexverse.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from lexverse.models.trial import (
    AgentConfig,
    EnvironmentType,
    MCPServerConfig,
    TaskEnvConfig,
    TrialPaths,
)
from lexverse.tasks.user import AgentLimits, OutputContract, RetrievalConfig, TaskCapabilities
from lexverse.verifiers.sandbox import SandboxVerifier


class DummyCustomLegalAgent(BaseInstalledAgent):
    """Custom agent for testing CLI flags, env vars, and sandbox execution."""

    SUPPORTS_ATIF = True

    CLI_FLAGS = [
        CliFlag("max_turns", cli="--max-turns", type="int", default=5),
        CliFlag("strict", cli="--strict", type="bool"),
    ]
    ENV_VARS = [
        EnvVar("api_key", env="CUSTOM_API_KEY", type="str", default="dummy-key"),
    ]

    @staticmethod
    @override
    def name() -> str:
        return "dummy-custom-legal-agent"

    @override
    def get_version_command(self) -> str | None:
        return "echo 1.2.3"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        return None

    @override
    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        await self.exec_as_agent(
            environment,
            command=(
                "(cat /inputs/contract.md > /workspace/draft.md 2>/dev/null || true) && "
                "echo 'Reviewed: compliant' > /artifacts/review.md"
            ),
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        traj = Trajectory(
            session_id=self.session_id or "test-session",
            agent=Agent(name=self.name(), version=self.version() or "1.2.3", model_name=self.model_name),
            steps=[
                Step(step_id=1, source="user", message="Review contract"),
                Step(
                    step_id=2,
                    source="agent",
                    message="Reading contract and writing review",
                    reasoning_content="Checking legal clauses in /inputs/contract.md",
                    tool_calls=[
                        ToolCall(
                            tool_call_id="call_1",
                            function_name="bash",
                            arguments={"command": "cat /inputs/contract.md"},
                        )
                    ],
                    observation=Observation(
                        results=[
                            ObservationResult(
                                source_call_id="call_1",
                                content="Article 1: Confidentiality",
                            )
                        ]
                    ),
                    metrics=Metrics(prompt_tokens=120, completion_tokens=45, cost_usd=0.002),
                ),
                Step(
                    step_id=3,
                    source="agent",
                    message="最终法律意见：经审查，该合同保密条款符合《民法典》规定，合规通过。",
                    metrics=Metrics(prompt_tokens=150, completion_tokens=30, cost_usd=0.001),
                ),
            ],
            final_metrics=FinalMetrics(
                total_prompt_tokens=270,
                total_completion_tokens=75,
                total_cost_usd=0.003,
                total_steps=3,
            ),
        )
        write_atif_trajectory(traj, self.logs_dir / "trajectory.json")
        context.n_input_tokens = 270
        context.n_output_tokens = 75
        context.cost_usd = 0.003


def test_atif_trajectory_validation_and_langchain_conversion(tmp_path: Path):
    messages = [
        {"role": "system", "content": "You are a legal assistant."},
        {"role": "user", "content": "Search company law article 84."},
        {
            "role": "assistant",
            "content": "Let me search the knowledge base.",
            "additional_kwargs": {"reasoning_content": "Need Company Law Art 84"},
            "tool_calls": [
                {
                    "id": "tc_001",
                    "name": "search_knowledge",
                    "args": {"query": "公司法 第八十四条"},
                }
            ],
            "usage_metadata": {"input_tokens": 200, "output_tokens": 50},
        },
        {
            "role": "tool",
            "tool_call_id": "tc_001",
            "content": "第八十四条 有限责任公司的股东之间可以相互转让其全部或者部分股权。",
        },
        {
            "role": "assistant",
            "content": "根据《公司法》第八十四条，股东之间可以自由转让股权。",
            "usage_metadata": {"input_tokens": 280, "output_tokens": 60},
        },
    ]

    traj = convert_langchain_messages_to_atif(
        messages,
        session_id="sess_01",
        agent_name="lexverse",
        model_name="openai/gpt-4o",
    )
    assert traj is not None
    assert traj.schema_version == "ATIF-v1.7"
    assert len(traj.steps) == 4
    assert traj.steps[2].tool_calls is not None
    assert traj.steps[2].tool_calls[0].function_name == "search_knowledge"
    assert traj.steps[2].observation is not None
    assert traj.steps[2].observation.results[0].source_call_id == "tc_001"
    assert traj.final_metrics is not None
    assert traj.final_metrics.total_prompt_tokens == 480
    assert traj.final_metrics.total_completion_tokens == 110

    out_file = tmp_path / "trajectory.json"
    write_atif_trajectory(traj, out_file)
    reloaded = Trajectory.model_validate_json(out_file.read_text(encoding="utf-8"))
    assert reloaded.session_id == "sess_01"


def test_agent_factory_supports_all_general_agents_and_custom_import(tmp_path: Path):
    supported = AgentFactory.supported_agent_names()
    for expected in (
        "lexverse",
        "claude-code",
        "codex",
        "qwen-coder",
        "gemini-cli",
        "openhands",
        "opencode",
        "aider",
        "kimi-cli",
        "goose",
        "cursor-cli",
        "mini-swe-agent",
        "pi",
        "cline-cli",
        "copilot-cli",
        "swe-agent",
        "trae-agent",
        "openhands-sdk",
        "deerflow",
        "hermes",
        "mimo",
        "openclaw",
        "vibe",
        "oracle",
        "nop",
    ):
        assert expected in supported
    assert "deepagents" not in supported
    assert "qwen-code" not in supported

    mcp_servers = [
        MCPServerConfig(
            name="lexverse_knowledge",
            transport="stdio",
            command="python",
            args=["-m", "lexverse.capabilities.knowledge_mcp_server", "--manifest", "/tmp/m.json"],
        ),
        MCPServerConfig(
            name="pkulaw_law_search",
            transport="sse",
            url="https://mcp.pkulaw.com/sse",
        ),
    ]

    claude = AgentFactory.create_agent_from_name(
        "claude-code",
        logs_dir=tmp_path / "claude",
        model_name="anthropic/claude-sonnet-4-5",
        mcp_servers=mcp_servers,
        skills_dir="/skills",
    )
    assert isinstance(claude, ClaudeCode)
    assert "/skills" in (claude._build_register_skills_command() or "")
    mcp_cmd = claude._build_register_mcp_servers_command() or ""
    assert "lexverse_knowledge" in mcp_cmd and "pkulaw_law_search" in mcp_cmd

    codex = AgentFactory.create_agent_from_name(
        "codex",
        logs_dir=tmp_path / "codex",
        model_name="openai/gpt-5",
        mcp_servers=mcp_servers,
        skills_dir="/skills",
    )
    assert isinstance(codex, Codex)
    assert "/skills" in (codex._build_register_skills_command() or "")

    qwen = AgentFactory.create_agent_from_name(
        "qwen-coder",
        logs_dir=tmp_path / "qwen",
        model_name="openai/qwen3-coder-plus",
        mcp_servers=mcp_servers,
        skills_dir="/skills",
    )
    assert isinstance(qwen, QwenCode)
    assert "lexverse_knowledge" in (qwen._build_register_mcp_servers_command() or "")

    gemini = AgentFactory.create_agent_from_name(
        "gemini-cli",
        logs_dir=tmp_path / "gemini",
        model_name="gemini/gemini-2.5-pro",
        mcp_servers=mcp_servers,
        skills_dir="/skills",
    )
    assert isinstance(gemini, GeminiCli)

    openhands = AgentFactory.create_agent_from_name(
        "openhands",
        logs_dir=tmp_path / "openhands",
        model_name="openai/gpt-4o",
        mcp_servers=mcp_servers,
        skills_dir="/skills",
    )
    assert isinstance(openhands, OpenHands)
    assert "lexverse_knowledge" in (openhands._build_mcp_config_toml() or "")

    opencode = AgentFactory.create_agent_from_name(
        "opencode",
        logs_dir=tmp_path / "opencode",
        model_name="openai/gpt-4o",
        mcp_servers=mcp_servers,
        skills_dir="/skills",
    )
    assert isinstance(opencode, OpenCode)

    # Custom import path via AgentConfig
    cfg = AgentConfig(
        name=f"{__name__}:DummyCustomLegalAgent",
        model_name="openai/gpt-4o",
        kwargs={"max_turns": 10, "strict": True},
    )
    custom = AgentFactory.create_agent_from_config(cfg, logs_dir=tmp_path / "custom")
    assert isinstance(custom, DummyCustomLegalAgent)
    assert custom.build_cli_flags() == "--max-turns 10 --strict"


def test_local_environment_and_custom_agent_and_verifier_end_to_end(tmp_path: Path):
    async def _run():
        trial_dir = tmp_path / "trial_01"
        trial_paths = TrialPaths(trial_dir=trial_dir)
        trial_paths.mkdir()

        # Seed /inputs and /skills
        (trial_paths.inputs_dir / "contract.md").write_text(
            "Article 1: Confidentiality", encoding="utf-8"
        )
        (trial_paths.skills_dir / "bindings.json").write_text(
            json.dumps({"tools": ["search_knowledge"]}), encoding="utf-8"
        )

        # Create tests/test.sh
        tests_dir = tmp_path / "task_tests"
        tests_dir.mkdir()
        (tests_dir / "test.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -e\n"
            "grep -q 'Reviewed: compliant' /artifacts/review.md\n"
            "echo '{\"reward\": 1.0, \"compliance_check\": 1.0}' > /logs/verifier/reward.json\n",
            encoding="utf-8",
        )

        env = EnvironmentFactory.create_environment(
            env_type=EnvironmentType.LOCAL,
            environment_dir=tmp_path / "env_def",
            environment_name="contract-review",
            session_id="sess_e2e",
            trial_paths=trial_paths,
            task_env_config=TaskEnvConfig(workdir="/workspace", skills_dir="/skills"),
        )
        assert isinstance(env, LocalEnvironment)

        await env.start()
        try:
            agent = DummyCustomLegalAgent(
                logs_dir=trial_paths.agent_dir,
                model_name="openai/gpt-4o",
                session_id="sess_e2e",
            )
            context = AgentContext()
            await agent.setup(env)
            await agent.run("Please review /inputs/contract.md", env, context)

            assert (trial_paths.artifacts_dir / "review.md").is_file()
            assert (trial_paths.workspace_dir / "draft.md").is_file()
            assert (trial_paths.agent_dir / "trajectory.json").is_file()
            assert context.n_input_tokens == 270

            verifier = SandboxVerifier(
                tests_dir=tests_dir,
                trial_paths=trial_paths,
                environment=env,
            )
            v_res = await verifier.verify()
            assert v_res.return_code == 0
            assert v_res.rewards == {"reward": 1.0, "compliance_check": 1.0}
        finally:
            await env.stop()

    asyncio.run(_run())


def test_build_mcp_server_configs_includes_local_knowledge_mcp(tmp_path: Path):
    spec = AgentExecutionSpec(
        run_dir=tmp_path,
        thread_id="t1",
        messages=[{"role": "user", "content": "hi"}],
        model=ModelConfig(name="gpt-4o", provider="openai_compatible", profile="openai"),
        selection=TaskCapabilities(
            knowledge=[tmp_path],
            mcp=["pkulaw_law_search"],
            tools=["search_knowledge", "fetch_knowledge"],
        ),
        retrieval=RetrievalConfig(mode="keyword"),
        limits=AgentLimits(),
        outputs=OutputContract(),
        interactive=False,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    servers = build_mcp_server_configs(spec, manifest_path, cache_dir=tmp_path / "cache")
    names = {s.name: s for s in servers}
    assert "lexverse_knowledge" in names
    assert names["lexverse_knowledge"].transport == "stdio"
    assert "pkulaw_law_search" in names
    assert names["pkulaw_law_search"].transport == "sse"


def test_knowledge_mcp_server_end_to_end_retrieval(tmp_path: Path):
    """Verify build_index + knowledge_mcp_server search_knowledge and fetch_knowledge work end-to-end."""
    from lexverse.capabilities.knowledge.index import build_index
    from lexverse.capabilities.knowledge_mcp_server import _load_retriever

    knowledge_dir = tmp_path / "legal_laws" / "民法典"
    knowledge_dir.mkdir(parents=True)
    (knowledge_dir / "contract_articles.md").write_text(
        "第五百零九条 当事人应当按照约定全面履行自己的义务。当事人应当遵循诚信原则，根据合同的性质、目的和交易习惯履行通知、协助、保密等义务。\n\n"
        "第五百八十五条 当事人可以约定一方违约时应当根据违约情况向对方支付一定数额的违约金，也可以约定因违约产生的损失赔偿额的计算方法。",
        encoding="utf-8",
    )

    cache_dir = tmp_path / "cache"
    retrieval_cfg = RetrievalConfig(mode="keyword", top_k=3, keyword_candidates=4)
    index_path = build_index(knowledge_dir, cache_dir, retrieval_cfg)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "resources": {
                    "knowledge": [
                        {"path": str(index_path), "index_hash": index_path.name,
                         "retrieval": retrieval_cfg.model_dump(mode="json")}
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    retriever = _load_retriever(manifest_path, cache_dir=cache_dir)
    assert next(iter(retriever.indexes.values())).config.top_k == 3
    assert next(iter(retriever.indexes.values())).config.keyword_candidates == 4
    hits = retriever.search("违约金 损失赔偿", top_k=3)
    assert len(hits) >= 1
    assert "第五百八十五条" in hits[0]["snippet"]
    citation_id = hits[0]["citation_id"]
    fetched = retriever.fetch(citation_id)
    assert "第五百八十五条" in fetched["text"]


def test_lexverse_task_run_cli_with_oracle_and_custom_agents(tmp_path: Path):
    task_dir = tmp_path / "contract_task"
    inputs_dir = task_dir / "inputs"
    solution_dir = task_dir / "solution"
    tests_dir = task_dir / "tests"
    inputs_dir.mkdir(parents=True)
    solution_dir.mkdir(parents=True)
    tests_dir.mkdir(parents=True)

    (inputs_dir / "contract.md").write_text("甲方：A公司\n乙方：B公司\n争议解决：提交北京仲裁委员会仲裁。", encoding="utf-8")

    # Oracle solve.sh
    (solution_dir / "solve.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "mkdir -p /artifacts\n"
        "echo '# 审查结论\nReviewed: compliant' > /artifacts/review.md\n",
        encoding="utf-8",
    )

    # Verifier test.sh
    (tests_dir / "test.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -e\n"
        "grep -q 'Reviewed: compliant' /artifacts/review.md\n"
        "echo '1.0' > /logs/verifier/reward.txt\n",
        encoding="utf-8",
    )

    task_yaml = task_dir / "task.yaml"
    task_yaml.write_text(
        "schema_version: 1\n"
        "id: contract_review_demo\n"
        "instruction: 审查输入合同并生成 /artifacts/review.md\n"
        "inputs: ./inputs\n"
        "model:\n"
        "  provider: openai_compatible\n"
        "  profile: openai\n"
        "  name: gpt-4o\n"
        "capabilities:\n"
        "  tools: [search_knowledge, fetch_knowledge]\n"
        "outputs:\n"
        "  required: [artifacts/review.md]\n"
        "  allowed_formats: [md, json, txt]\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "runs"

    # 1. Run with --agent oracle --env local
    rc_oracle = task_cli_main(
        [
            "run",
            str(task_yaml),
            "--output-root",
            str(output_root),
            "--no-interactive",
            "--agent",
            "oracle",
            "--env",
            "local",
        ]
    )
    assert rc_oracle == 0

    # 2. Run with --agent <custom class> --env local
    rc_custom = task_cli_main(
        [
            "run",
            str(task_yaml),
            "--output-root",
            str(output_root),
            "--no-interactive",
            "--agent",
            f"{__name__}:DummyCustomLegalAgent",
            "--env",
            "local",
        ]
    )
    assert rc_custom == 0

    run_dirs = list((output_root / "contract_review_demo").iterdir())
    assert len(run_dirs) == 2
    by_agent = {}
    for d in run_dirs:
        res_data = json.loads((d / "result.json").read_text(encoding="utf-8"))
        by_agent[res_data["agent"]] = (d, res_data)

    assert "oracle" in by_agent
    assert "dummy-custom-legal-agent" in by_agent

    custom_run, res_data = by_agent["dummy-custom-legal-agent"]
    assert res_data["status"] == "completed"
    assert res_data["answer"] == "最终法律意见：经审查，该合同保密条款符合《民法典》规定，合规通过。"
    assert res_data["verifier"]["rewards"] == {"reward": 1.0}
    assert (custom_run / "trajectory.json").is_file()


def test_benchmark_runner_and_plawbench_with_general_agent(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LEXVERSE_OPENAI_API_KEY", "sk-test-dummy-key")
    monkeypatch.setenv("LEXVERSE_OPENAI_BASE_URL", "https://api.openai.com/v1")

    from lexverse.agents.benchmark_runner import execute as benchmark_execute
    from lexverse.capabilities.benchmark import parse_capabilities

    # Verify PLawBench is now accepted by parse_capabilities alongside LawBench and LexEval
    cap_cfg = parse_capabilities(
        {
            "enabled": True,
            "agent": f"{__name__}:DummyCustomLegalAgent",
            "environment": "local",
            "defaults": {"skills": [], "knowledge": [], "mcp": []},
        },
        "plawbench",
        ModelConfig(provider="openai_compatible", profile="openai", name="gpt-4o-mini"),
        tmp_path,
    )
    assert cap_cfg["enabled"] is True
    assert cap_cfg["agent"] == f"{__name__}:DummyCustomLegalAgent"

    async def _run_benchmark_flow():
        template_dir = tmp_path / "template_case_analysis"
        # Seed a dummy input so DummyCustomLegalAgent's `cat /inputs/contract.md` succeeds
        prep_payload = {
            "operation": "prepare",
            "run_dir": str(template_dir),
            "task_id": "case_analysis",
            "task_type": "case_analysis",
            "capabilities": cap_cfg,
            "model": {
                "provider": "openai_compatible",
                "profile": "openai",
                "name": "gpt-4o-mini",
            },
        }
        prep_res = await benchmark_execute(prep_payload)
        assert prep_res["status"] == "completed"
        assert (template_dir / ".internal/manifest.json").is_file()

        trial_dir = tmp_path / "trial_case_01"
        exec_payload = {
            "operation": "execute",
            "run_dir": str(trial_dir),
            "template": str(template_dir),
            "task_id": "plawbench:case_analysis:1",
            "task_type": "case_analysis",
            "messages": [{"role": "user", "content": "请分析本案的保密条款效力。"}],
            "capabilities": cap_cfg,
            "model": {
                "provider": "openai_compatible",
                "profile": "openai",
                "name": "gpt-4o-mini",
            },
        }
        exec_res = await benchmark_execute(exec_payload)
        assert exec_res["status"] == "completed"
        # Verify the final answer was extracted from the ATIF trajectory for benchmark scoring!
        assert (
            exec_res["answer"]
            == "最终法律意见：经审查，该合同保密条款符合《民法典》规定，合规通过。"
        )
        assert (trial_dir / "trajectory.json").is_file()

    asyncio.run(_run_benchmark_flow())


def test_docker_environment_with_custom_agent_and_verifier(tmp_path: Path):
    """Verify DockerEnvironment runs a task inside an isolated container and runs SandboxVerifier."""
    import shutil
    import subprocess

    if not shutil.which("docker"):
        return
    info = subprocess.run(["docker", "info"], capture_output=True, text=True, check=False)
    if info.returncode != 0:
        return

    async def _run():
        trial_dir = tmp_path / "trial_docker_01"
        trial_paths = TrialPaths(trial_dir=trial_dir)
        trial_paths.mkdir()

        (trial_paths.inputs_dir / "contract.md").write_text(
            "Article 1: Confidentiality", encoding="utf-8"
        )
        (trial_paths.skills_dir / "bindings.json").write_text(
            json.dumps({"tools": ["search_knowledge"]}), encoding="utf-8"
        )

        tests_dir = tmp_path / "task_tests_docker"
        tests_dir.mkdir()
        (tests_dir / "test.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -e\n"
            "grep -q 'Reviewed: compliant' /artifacts/review.md\n"
            "echo '{\"reward\": 1.0, \"docker_isolated\": 1.0}' > /logs/verifier/reward.json\n",
            encoding="utf-8",
        )

        env = EnvironmentFactory.create_environment(
            env_type=EnvironmentType.DOCKER,
            environment_dir=tmp_path / "env_def_docker",
            environment_name="contract-review-docker",
            session_id="sess_docker_e2e",
            trial_paths=trial_paths,
            task_env_config=TaskEnvConfig(
                workdir="/workspace",
                skills_dir="/skills",
                docker_image="python:3.12-slim",
            ),
        )

        await env.start()
        try:
            agent = DummyCustomLegalAgent(
                logs_dir=trial_paths.agent_dir,
                model_name="openai/gpt-4o",
                session_id="sess_docker_e2e",
            )
            context = AgentContext()
            await agent.setup(env)
            await agent.run("Please review /inputs/contract.md", env, context)

            assert (trial_paths.artifacts_dir / "review.md").is_file()
            assert (trial_paths.workspace_dir / "draft.md").is_file()
            assert (trial_paths.agent_dir / "trajectory.json").is_file()

            verifier = SandboxVerifier(
                tests_dir=tests_dir,
                trial_paths=trial_paths,
                environment=env,
            )
            v_res = await verifier.verify()
            assert v_res.return_code == 0
            assert v_res.rewards == {"reward": 1.0, "docker_isolated": 1.0}
        finally:
            await env.stop()

    asyncio.run(_run())


def test_shipped_contract_review_task_and_inspect_cli(tmp_path: Path, capsys, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    example_task = repo_root / "user_workspace/task/example/task.yaml"
    assert example_task.is_file()
    assert (repo_root / "user_workspace/task/example/inputs/service_agreement.md").is_file()

    # 1. Verify `lexverse task agents` CLI lists all supported agents
    rc_agents = task_cli_main(["agents"])
    assert rc_agents == 0
    agents_out = json.loads(capsys.readouterr().out)
    assert "claude-code" in agents_out["supported_agents"]
    assert "lexverse" in agents_out["supported_agents"]

    # 1b. Verify `lexverse task init <task_name>` interactive scaffolding
    replies = iter(["y", "y", "y"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(replies))
    rc_init = task_cli_main(["init", "my_new_task", "--root", str(tmp_path / "tasks_root")])
    assert rc_init == 0
    init_out = json.loads(capsys.readouterr().out)
    init_dir = Path(init_out["task_dir"])
    assert (init_dir / "task.yaml").is_file()
    assert (init_dir / "inputs").is_dir()
    assert (init_dir / "environment/Dockerfile").is_file()
    assert (init_dir / "solution/solve.sh").is_file()
    assert (init_dir / "tests/test.sh").is_file()

    # 2. Run the shipped `user_workspace/task/example/task.yaml` with `--agent oracle`
    output_root = tmp_path / "example_runs"
    rc_oracle = task_cli_main(
        [
            "run",
            str(example_task),
            "--output-root",
            str(output_root),
            "--no-interactive",
            "--agent",
            "oracle",
            "--env",
            "local",
        ]
    )
    assert rc_oracle == 0
    oracle_out = json.loads(capsys.readouterr().out)
    assert oracle_out["status"] == "completed"
    assert oracle_out["verifier"]["rewards"]["reward"] == 1.0
    assert oracle_out["verifier"]["rewards"]["all_pass_score"] == 1.0
    assert oracle_out["verifier"]["rewards"]["passed_criteria"] == 4.0
    assert oracle_out["verifier"]["rewards"]["total_criteria"] == 4.0

    oracle_run_dir = Path(oracle_out["run_dir"])
    assert (oracle_run_dir / "artifacts/review_memo.md").is_file()

    # 3. Update outputs.required temporarily or use a custom task to test DummyCustomLegalAgent + summarize_atif_trajectory
    custom_task_dir = tmp_path / "custom_inspect_task"
    (custom_task_dir / "inputs").mkdir(parents=True)
    (custom_task_dir / "inputs/contract.md").write_text("Confidentiality", encoding="utf-8")
    custom_task_yaml = custom_task_dir / "task.yaml"
    custom_task_yaml.write_text(
        "schema_version: 1\n"
        "id: custom_inspect_task\n"
        "instruction: Review contract\n"
        "inputs: ./inputs\n"
        "model:\n"
        "  provider: openai_compatible\n"
        "  profile: openai\n"
        "  name: gpt-4o\n"
        "outputs:\n"
        "  required: [artifacts/review.md]\n",
        encoding="utf-8",
    )
    task_cli_main(
        [
            "run",
            str(custom_task_yaml),
            "--output-root",
            str(output_root),
            "--no-interactive",
            "--agent",
            f"{__name__}:DummyCustomLegalAgent",
            "--env",
            "local",
        ]
    )
    custom_out = json.loads(capsys.readouterr().out)
    run_dir = Path(custom_out["run_dir"])
    assert (run_dir / "trajectory.json").is_file()

    from lexverse.commands.tasks import summarize_atif_trajectory
    summary = summarize_atif_trajectory(run_dir / "trajectory.json")
    assert summary["schema_version"] == "ATIF-v1.7"
    assert summary["total_steps"] == 3
    assert summary["total_tool_calls"] == 1
    assert summary["tool_calls_by_name"] == {"bash": 1}




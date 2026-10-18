from __future__ import annotations

import json
from pathlib import Path
from typing import Any, override

from lexverse.agents.base import BaseAgent
from lexverse.capabilities.base import AgentExecutionSpec
from lexverse.environments.base import BaseEnvironment
from lexverse.models.agent.context import AgentContext
from lexverse.models.agent.name import AgentName


class LexVerseAgent(BaseAgent):
    """
    Built-in LexVerse LangGraph/DeepAgents implementation wrapped as a standard
    Harbor-compatible BaseAgent.
    """

    SUPPORTS_ATIF: bool = True
    SUPPORTS_RESUME: bool = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        spec: AgentExecutionSpec | None = None,
        reply: str | None = None,
        confirm_facts: bool = False,
        model: Any | None = None,
        cache: Path | None = None,
        adapter_factory: Any | None = None,
        prepare_only: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, *args, **kwargs)
        self.spec = spec
        self.reply = reply
        self.confirm_facts = confirm_facts
        self.model = model
        self.cache = cache
        self.adapter_factory = adapter_factory
        self.prepare_only = prepare_only
        self.last_result: dict[str, Any] | None = None

    @staticmethod
    @override
    def name() -> str:
        return AgentName.LEXVERSE.value

    @override
    def version(self) -> str:
        return "0.2.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self.spec is None:
            raise ValueError("LexVerseAgent requires an AgentExecutionSpec (`spec`).")

        from lexverse.agents.agent_loop import execute_agent

        result = await execute_agent(
            self.spec,
            reply=self.reply,
            confirm_facts=self.confirm_facts,
            model=self.model,
            cache=self.cache,
            adapter_factory=self.adapter_factory,
            prepare_only=self.prepare_only,
        )
        self.last_result = result
        usage = result.get("usage") or {}
        context.n_input_tokens = usage.get("input_tokens")
        context.n_output_tokens = usage.get("output_tokens")
        context.metadata = {"lexverse_result": result}

        # Also copy ATIF trajectory.json into logs_dir if logs_dir != run_dir
        traj_src = self.spec.run_dir.resolve() / "trajectory.json"
        traj_dst = self.logs_dir / "trajectory.json"
        if traj_src.exists() and traj_src.resolve() != traj_dst.resolve():
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            traj_dst.write_text(traj_src.read_text(encoding="utf-8"), encoding="utf-8")

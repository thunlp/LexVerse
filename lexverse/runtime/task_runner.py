from __future__ import annotations

from pathlib import Path

from lexverse.capabilities.base import CapabilitySelection
from lexverse.capabilities.runtime import CapabilityRuntime
from lexverse.environments.base import ExecutionEnvironment
from lexverse.environments.direct_response import DirectResponseEnvironment
from lexverse.interaction.engine import InteractionEngine
from lexverse.interaction.participants import Participant
from lexverse.interaction.policies import InteractionPolicy
from lexverse.interaction.schema import EnvironmentResult
from lexverse.tasks.schema import LexVerseTask


class TaskRunner:
    def __init__(self) -> None:
        self.interaction = InteractionEngine()
        self.capabilities = CapabilityRuntime()

    async def run_environment(
        self,
        *,
        task: LexVerseTask,
        policy: InteractionPolicy | None,
        participants: dict[str, Participant] | None,
        work_dir: Path,
        capability_selection: CapabilitySelection | None = None,
        environment: ExecutionEnvironment | None = None,
    ) -> EnvironmentResult:
        """Run the interaction phase without forcing per-Trial verification."""
        capability_context = await self.capabilities.prepare(
            capability_selection or CapabilitySelection()
        )
        if environment is not None:
            await environment.prepare({"capabilities": capability_context})
            env_result = await environment.run(task, work_dir)
        else:
            if policy is None or participants is None:
                raise ValueError(
                    "an environment or direct-response policy and participants are required"
                )
            native_environment = DirectResponseEnvironment(
                policy=policy,
                participants=participants,
                context={"capabilities": capability_context},
                engine=self.interaction,
            )
            env_result = await native_environment.run(task, work_dir)
        return env_result

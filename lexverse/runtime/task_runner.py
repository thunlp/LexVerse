from __future__ import annotations

from pathlib import Path

from lexverse.capabilities.base import CapabilitySelection
from lexverse.capabilities.runtime import CapabilityRuntime
from lexverse.environments.base import ExecutionEnvironment
from lexverse.environments.direct_response import DirectResponseEnvironment
from lexverse.interaction.engine import InteractionEngine
from lexverse.interaction.participants.base import Participant
from lexverse.interaction.policies.base import InteractionPolicy
from lexverse.interaction.result import EnvironmentResult
from lexverse.tasks.schema import LexVerseTask
from lexverse.verifiers.base import TaskVerifier
from lexverse.verifiers.result import VerifierResult
from lexverse.verifiers.runtime import VerifierRuntime


class TaskRunResult:
    def __init__(self, environment: EnvironmentResult, verifier: VerifierResult):
        self.environment = environment
        self.verifier = verifier


class TaskRunner:
    def __init__(self) -> None:
        self.interaction = InteractionEngine()
        self.verification = VerifierRuntime()
        self.capabilities = CapabilityRuntime()

    async def run(
        self,
        *,
        task: LexVerseTask,
        policy: InteractionPolicy | None,
        participants: dict[str, Participant] | None,
        verifier: TaskVerifier,
        work_dir: Path,
        verifier_context: dict | None = None,
        capability_selection: CapabilitySelection | None = None,
        environment: ExecutionEnvironment | None = None,
    ) -> TaskRunResult:
        env_result = await self.run_environment(
            task=task,
            policy=policy,
            participants=participants,
            work_dir=work_dir,
            capability_selection=capability_selection,
            environment=environment,
        )
        verified = await self.verification.run(
            verifier, task, env_result, work_dir / "verifier",
            context=verifier_context,
        )
        return TaskRunResult(env_result, verified)

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

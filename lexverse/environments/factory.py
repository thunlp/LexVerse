from __future__ import annotations

from pathlib import Path

from lexverse.environments.base import ExecutionEnvironment
from lexverse.environments.docker import DockerEnvironment
from lexverse.environments.local import LocalEnvironment
from lexverse.models.task.environment import EnvironmentType, TaskEnvConfig
from lexverse.models.trial.paths import TrialPaths
from lexverse.utils.import_path import import_class


class EnvironmentFactory:
    _ENV_MAP: dict[EnvironmentType, type[ExecutionEnvironment]] = {
        EnvironmentType.LOCAL: LocalEnvironment,
        EnvironmentType.DOCKER: DockerEnvironment,
    }

    @classmethod
    def create_environment(
        cls,
        env_type: EnvironmentType | str,
        environment_dir: Path,
        environment_name: str,
        session_id: str,
        trial_paths: TrialPaths,
        task_env_config: TaskEnvConfig,
        **kwargs,
    ) -> ExecutionEnvironment:
        if isinstance(env_type, str) and ":" in env_type:
            env_cls = import_class(env_type, base=ExecutionEnvironment, label="environment")
            return env_cls(
                environment_dir=environment_dir,
                environment_name=environment_name,
                session_id=session_id,
                trial_paths=trial_paths,
                task_env_config=task_env_config,
                **kwargs,
            )

        resolved_type = EnvironmentType(env_type)
        env_cls = cls._ENV_MAP[resolved_type]
        if (environment_dir / "docker-compose.yaml").is_file():
            if resolved_type != EnvironmentType.DOCKER:
                raise ValueError("Compose tasks require Docker")
            from lexverse.environments.compose import ComposeEnvironment
            env_cls = ComposeEnvironment
        return env_cls(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

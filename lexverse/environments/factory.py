from __future__ import annotations

from pathlib import Path

from lexverse.environments.base import BaseEnvironment
from lexverse.environments.docker import DockerEnvironment
from lexverse.environments.local import LocalEnvironment
from lexverse.models.trial import EnvironmentType, TaskEnvConfig, TrialPaths
from lexverse.utils.import_path import import_class


class EnvironmentFactory:
    _ENV_MAP: dict[EnvironmentType, type[BaseEnvironment]] = {
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
    ) -> BaseEnvironment:
        if isinstance(env_type, str) and ":" in env_type:
            env_cls = import_class(env_type, base=BaseEnvironment, label="environment")
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
        return env_cls(
            environment_dir=environment_dir,
            environment_name=environment_name,
            session_id=session_id,
            trial_paths=trial_paths,
            task_env_config=task_env_config,
            **kwargs,
        )

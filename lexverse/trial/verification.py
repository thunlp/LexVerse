# Copyright 2025-2026 Alex Shaw and the Harbor Framework Team
# Portions of this file are derived from Harbor (https://github.com/laude-institute/harbor, v0.20.0).
# Modified by the LexVerse Contributors for LexVerse package integration and legal capability support.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path
import json

from lexverse.models.trial.paths import TrialPaths
from lexverse.verifiers.task import TaskVerifier


async def verify_trial(spec, agent_environment, tests_dir: Path, records):
    """Create a fresh scoring instance when requested; never fall back to shared."""
    from lexverse.environments.factory import EnvironmentFactory
    from lexverse.models.task.environment import TaskEnvConfig
    from lexverse.trial.artifacts import restore

    config = spec.verifier
    if config is None:
        return None
    separate = config["environment_mode"] == "separate"
    environment = agent_environment
    paths = agent_environment.trial_paths if agent_environment is not None else TrialPaths(spec.run_dir)
    upload_tests = True
    phase = "verification"
    try:
        verifier_env = {}
        if config.get("model"):
            if not separate:
                raise ValueError("Judge credentials require separate verification")
            from lexverse.agents.model_connection import ModelConnectionSpec, resolve_model_connection

            model = config["model"]
            if model["provider"] not in {"openai", "openai_compatible"}:
                raise ValueError("User Task Judge requires an OpenAI-compatible provider")
            connection = resolve_model_connection(
                model["name"], ModelConnectionSpec(provider="openai", prefixed_model=False),
                profile=model.get("profile"))
            if not connection.api_key or not connection.base_url:
                raise ValueError("Judge connection requires an API key and base URL")
            verifier_env = {
                "LEXVERSE_JUDGE_MODEL": connection.model,
                "LEXVERSE_JUDGE_BASE_URL": connection.base_url,
                "LEXVERSE_JUDGE_API_KEY": connection.api_key,
                "LEXVERSE_JUDGE_PARAMETERS": json.dumps(model.get("generation_parameters", {})),
                "LEXVERSE_TASK_INSTRUCTION": "\n\n".join(message["content"] for message in spec.messages),
            }
        if separate:
            if spec.environment != "docker":
                raise ValueError("Separate verification requires Docker")
            phase = "verifier_environment_start"
            dedicated = config.get("environment") or {}
            dedicated_image = dedicated.get("docker_image")
            test_build = (tests_dir / "Dockerfile").is_file() or (tests_dir / "docker-compose.yaml").is_file()
            agent_definition = (agent_environment.environment_dir if agent_environment is not None
                                else tests_dir.parent / 'environment')
            definition = (spec.run_dir / ".internal/scoring/image" if dedicated_image else
                          tests_dir if test_build else agent_definition)
            options = dict(dedicated) if config.get("environment") is not None else dict(spec.environment_config)
            # Do not inherit agent credentials or MCP connections into scoring.
            options.pop("mcp_servers", None)
            options.pop("skills", None)
            options.pop("knowledge", None)
            if config.get("environment") is None:
                options.pop("env", None)
            if test_build and not dedicated_image:
                options["docker_image"] = None
            elif not dedicated_image:
                options["docker_image"] = spec.environment_config.get("docker_image")
            upload_tests = not (dedicated_image or test_build)
            paths = TrialPaths(spec.run_dir / ".internal/scoring")
            environment = EnvironmentFactory.create_environment(
                env_type="docker", environment_dir=definition, environment_name="verifier",
                session_id=spec.thread_id + "-verifier", trial_paths=paths,
                task_env_config=TaskEnvConfig(**options))
            await environment.start()
            phase = "artifact_restore"
            await restore(environment, records, spec.run_dir / "artifacts")
        phase = "verification"
        verifier = TaskVerifier(tests_dir, paths, environment,
                                   timeout_sec=config["timeout_sec"], upload_tests=upload_tests,
                                   env=verifier_env)
        return (await verifier.verify()).model_dump(mode="json")
    except Exception as exc:
        return {"rewards": None, "return_code": 1, "error_type": type(exc).__name__,
                "error_message": str(exc), "phase": phase}
    finally:
        if separate and environment is not agent_environment:
            await environment.stop()

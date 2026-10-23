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

import json
from pathlib import Path
import shlex
from typing import Any

from pydantic import BaseModel

from lexverse.environments.base import ExecutionEnvironment
from lexverse.models.trial import EnvironmentPaths, TrialPaths


class SandboxVerifierResult(BaseModel):
    rewards: dict[str, float | int] | None = None
    return_code: int = 0
    stdout: str | None = None
    stderr: str | None = None


class SandboxVerifier:
    """
    Harbor-compatible test script verifier.

    Uploads `tests/` into `/tests` in the sandbox environment, executes
    `test.sh`, and parses `/logs/verifier/reward.txt` or `/logs/verifier/reward.json`.
    """

    def __init__(
        self,
        tests_dir: Path,
        trial_paths: TrialPaths,
        environment: ExecutionEnvironment,
        timeout_sec: float | None = 300.0,
        env: dict[str, str] | None = None,
    ):
        self.tests_dir = tests_dir
        self.trial_paths = trial_paths
        self.environment = environment
        self.timeout_sec = int(timeout_sec) if timeout_sec is not None else None
        self.env = env or {}

    @property
    def test_script_path(self) -> Path:
        return self.tests_dir / "test.sh"

    def has_tests(self) -> bool:
        return self.tests_dir.is_dir() and self.test_script_path.is_file()

    def _parse_reward_text(self) -> dict[str, float | int]:
        content = self.trial_paths.reward_text_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError("Reward file is empty")
        return {"reward": float(content)}

    def _parse_reward_json(self) -> dict[str, float | int]:
        content = self.trial_paths.reward_json_path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError("Reward JSON file is empty")
        parsed: dict[str, Any] = json.loads(content)
        return {k: float(v) for k, v in parsed.items()}

    async def verify(self) -> SandboxVerifierResult:
        if not self.has_tests():
            return SandboxVerifierResult(rewards=None, return_code=0)

        env_paths = EnvironmentPaths.for_os(self.environment.os)
        await self.environment.upload_dir(
            source_dir=self.tests_dir,
            target_dir=str(env_paths.tests_dir),
        )
        container_test_sh = str(env_paths.tests_dir / "test.sh")
        container_stdout = str(env_paths.verifier_dir / "test-stdout.txt")

        await self.environment.exec(
            command=f"chmod +x {shlex.quote(container_test_sh)}",
            user=None if self.environment.type().value == "local" else "root",
        )

        exec_env = {"DEBIAN_FRONTEND": "noninteractive", **self.env}
        command = (
            f"bash {shlex.quote(container_test_sh)} "
            f"> {shlex.quote(container_stdout)} 2>&1"
        )
        res = await self.environment.exec(
            command=command,
            env=exec_env,
            timeout_sec=self.timeout_sec,
            user=None if self.environment.type().value == "local" else "root",
        )

        if not self.environment.is_mounted:
            for remote, local in (
                (str(env_paths.reward_text_path), self.trial_paths.reward_text_path),
                (str(env_paths.reward_json_path), self.trial_paths.reward_json_path),
                (container_stdout, self.trial_paths.test_stdout_path),
            ):
                try:
                    await self.environment.download_file(remote, local)
                except Exception:
                    pass

        rewards: dict[str, float | int] | None = None
        if self.trial_paths.reward_json_path.exists():
            try:
                rewards = self._parse_reward_json()
            except Exception:
                pass
        elif self.trial_paths.reward_text_path.exists():
            try:
                rewards = self._parse_reward_text()
            except Exception:
                pass

        stdout_text = (
            self.trial_paths.test_stdout_path.read_text(encoding="utf-8", errors="replace")
            if self.trial_paths.test_stdout_path.exists()
            else res.stdout
        )
        return SandboxVerifierResult(
            rewards=rewards,
            return_code=res.return_code,
            stdout=stdout_text,
            stderr=res.stderr,
        )

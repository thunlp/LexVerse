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

from datetime import datetime
from uuid import uuid4
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lexverse.models.task.environment import TaskOS


@dataclass(frozen=True)
class EnvironmentPaths:
    """
    Static paths used within the environment.
    """

    logs_dir: PurePosixPath = PurePosixPath("/logs")
    agent_dir: PurePosixPath = logs_dir / "agent"
    verifier_dir: PurePosixPath = logs_dir / "verifier"
    artifacts_dir: PurePosixPath = PurePosixPath("/artifacts")
    workspace_dir: PurePosixPath = PurePosixPath("/workspace")
    tests_dir: PurePosixPath = PurePosixPath("/tests")
    solution_dir: PurePosixPath = PurePosixPath("/solution")
    default_skills_dir: PurePosixPath = PurePosixPath("/skills")
    reward_text_path: PurePosixPath = verifier_dir / "reward.txt"
    reward_json_path: PurePosixPath = verifier_dir / "reward.json"

    @classmethod
    def for_windows(cls) -> "EnvironmentPaths":
        return cls._with_root(PurePosixPath("C:/"))

    @classmethod
    def for_os(cls, os: TaskOS) -> "EnvironmentPaths":
        if os == "windows":
            return cls.for_windows()
        return cls()

    @classmethod
    def _with_root(cls, root: PurePosixPath) -> "EnvironmentPaths":
        logs_dir = root / "logs"
        verifier_dir = logs_dir / "verifier"
        return cls(
            logs_dir=logs_dir,
            agent_dir=logs_dir / "agent",
            verifier_dir=verifier_dir,
            artifacts_dir=root / "artifacts",
            workspace_dir=root / "workspace",
            tests_dir=root / "tests",
            solution_dir=root / "solution",
            default_skills_dir=root / "skills",
            reward_text_path=verifier_dir / "reward.txt",
            reward_json_path=verifier_dir / "reward.json",
        )


@dataclass(frozen=True)
class TrialPaths:
    """
    Host output directory paths for a trial / task run.
    """

    trial_dir: Path

    @property
    def _agent_fs(self) -> Path:
        return self.trial_dir / ".internal" / "agent_fs"

    def mkdir(self, *, skills_root: Path | None = None) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.verifier_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        (skills_root or self.skills_dir).mkdir(parents=True, exist_ok=True)

    def chmod_dir(self) -> None:
        for path in (
            self.trial_dir,
            self.agent_dir,
            self.verifier_dir,
            self.artifacts_dir,
            self.workspace_dir,
        ):
            if path.exists():
                try:
                    path.chmod(0o777)
                except OSError:
                    pass

    @property
    def runtime_dir(self) -> Path:
        return self.trial_dir / ".internal" / "runtime"

    @property
    def agent_dir(self) -> Path:
        return self.runtime_dir / "agent"

    @property
    def artifacts_dir(self) -> Path:
        return self.trial_dir / "artifacts"

    @property
    def workspace_dir(self) -> Path:
        return self._agent_fs / "workspace"

    @property
    def skills_dir(self) -> Path:
        return self._agent_fs / "skills"

    @property
    def verifier_dir(self) -> Path:
        return self.trial_dir / ".internal" / "verifier"

    @property
    def test_stdout_path(self) -> Path:
        return self.verifier_dir / "test-stdout.txt"

    @property
    def test_stderr_path(self) -> Path:
        return self.verifier_dir / "test-stderr.txt"

    @property
    def reward_text_path(self) -> Path:
        return self.verifier_dir / "reward.txt"

    @property
    def reward_json_path(self) -> Path:
        return self.verifier_dir / "reward.json"

    @property
    def result_path(self) -> Path:
        return self.trial_dir / "result.json"

    @property
    def log_path(self) -> Path:
        return self.trial_dir / "trial.log"




def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:4]


class RunPaths:
    def __init__(self, run: Path):
        self.run = run
        self.internal = run / ".internal"
        self.agent_fs = self.internal / "agent_fs"
        self.artifacts = run / "artifacts"
        self.workspace = self.agent_fs / "workspace"

    @property
    def worker_result(self):
        return self.internal / 'control/outcome.json'

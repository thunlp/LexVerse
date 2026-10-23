from datetime import datetime
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES_ROOT = PROJECT_ROOT / ".lexverse/capabilities"
AGENT_RUNTIME_ROOT = PROJECT_ROOT / ".lexverse/runtime/agent"


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:4]


class RunPaths:
    def __init__(self, run: Path):
        self.run = run
        self.compact = (run / ".internal" / "agent_fs").is_dir()
        self.internal = run / ".internal" if self.compact else run
        self.agent_fs = self.internal / "agent_fs"
        self.inputs = self.agent_fs / "inputs" if self.compact else run / "inputs"
        self.artifacts = run / "artifacts" if self.compact else self.agent_fs / "artifacts"
        self.workspace = run / "workspace" if (run / "workspace").is_dir() else self.agent_fs / "workspace"

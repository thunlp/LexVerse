from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES_ROOT = PROJECT_ROOT / ".lexverse/capabilities"
AGENT_RUNTIME_ROOT = PROJECT_ROOT / ".lexverse/runtime/agent"


class RunPaths:
    def __init__(self, run: Path):
        self.run = run
        self.compact = (run / ".internal").is_dir()
        self.internal = run / ".internal" if self.compact else run
        self.agent_fs = self.internal / "agent_fs"
        self.inputs = self.agent_fs / "inputs" if self.compact else run / "inputs"
        self.artifacts = run / "artifacts" if self.compact else self.agent_fs / "artifacts"
        self.workspace = run / "workspace" if (run / "workspace").is_dir() else self.agent_fs / "workspace"

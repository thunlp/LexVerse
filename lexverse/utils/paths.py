"""Local installation and cache locations; no optional dependencies."""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES_ROOT = PROJECT_ROOT / ".lexverse/capabilities"
AGENT_RUNTIME_ROOT = PROJECT_ROOT / ".lexverse/runtime/agent"

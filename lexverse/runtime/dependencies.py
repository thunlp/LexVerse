import importlib.metadata
import json
from pathlib import Path
import platform
import sys

from lexverse.runtime.paths import AGENT_RUNTIME_ROOT


def dependency_root() -> Path:
    return AGENT_RUNTIME_ROOT


def is_macos_arm64() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def requires_agent_dependencies(agent_name, knowledge=(), inputs=None) -> bool:
    """Select the capability bundle from the work performed by LexVerse itself."""
    if agent_name == "lexverse" or knowledge:
        return True
    # The other input parsers have standard-library implementations/fallbacks.
    return inputs is not None and any(path.suffix.lower() == ".pdf" for path in Path(inputs).rglob("*"))


def activate_dependencies() -> dict:
    root = dependency_root()
    if not (root / "runtime.json").is_file():
        raise RuntimeError("Install Agent dependencies using: lexverse environment setup")

    try:
        manifest = json.loads((root / "runtime.json").read_text())
        if (not isinstance(manifest["python"], str) or not manifest["python"]
                or not isinstance(manifest["python_version"], list)
                or len(manifest["python_version"]) != 2
                or any(type(value) is not int for value in manifest["python_version"])
                or not isinstance(manifest["packages"], dict)
                or any(not isinstance(name, str) or not isinstance(version, str) or not name or not version
                       for name, version in manifest["packages"].items())):
            raise ValueError("Invalid manifest fields")
    except (OSError, ValueError, KeyError, TypeError):
        raise RuntimeError("Invalid Agent dependency manifest; rerun the installer") from None

    if list(sys.version_info[:2]) != manifest["python_version"]:
        raise RuntimeError("Agent dependencies were installed for another Python version")
    if Path(sys.executable).resolve() != Path(manifest["python"]).resolve():
        raise RuntimeError("Agent dependencies were installed with another Python interpreter; rerun the installer")

    if is_macos_arm64() and manifest.get("environment") != "conda":
        raise RuntimeError("Rebuild macOS arm64 Agent dependencies using: lexverse environment setup")

    if manifest.get("environment") == "conda" and not is_macos_arm64():
        raise RuntimeError("These Agent dependencies require macOS arm64")

    if any(name in sys.modules for name in ("pydantic", "numpy", "mcp", "langchain", "anthropic")):
        raise RuntimeError("Activate Agent dependencies before importing third-party modules")

    sys.path.insert(0, str(root / "site-packages"))

    for name, expected in manifest["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f"Agent dependency is missing: {name}") from None
        if actual != expected:
            raise RuntimeError(f"Agent dependency version mismatch: {name}")

    return manifest

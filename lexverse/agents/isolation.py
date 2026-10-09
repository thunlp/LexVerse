import importlib.metadata
import json
from pathlib import Path
import sys


def dependency_root() -> Path:
    return Path(__file__).resolve().parents[2] / ".lexverse/runtime/stage2"


def activate_dependencies() -> dict:
    root = dependency_root()
    if not (root / "runtime.json").is_file():
        raise RuntimeError("Install Stage2 dependencies using: lexverse environment setup")
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
        raise RuntimeError("Invalid Stage2 dependency manifest; rerun the installer") from None
    if list(sys.version_info[:2]) != manifest["python_version"]:
        raise RuntimeError("Stage2 dependencies were installed for another Python version")
    if Path(sys.executable).resolve() != Path(manifest["python"]).resolve():
        raise RuntimeError("Stage2 dependencies were installed with another Python interpreter; rerun the installer")
    if any(name in sys.modules for name in ("pydantic", "numpy", "mcp", "langchain", "anthropic")):
        raise RuntimeError("Activate Stage2 dependencies before importing third-party modules")
    sys.path.insert(0, str(root / "site-packages"))
    for name, expected in manifest["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(f"Stage2 dependency is missing: {name}") from None
        if actual != expected:
            raise RuntimeError(f"Stage2 dependency version mismatch: {name}")
    return manifest

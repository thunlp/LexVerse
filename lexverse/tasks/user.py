from pathlib import Path
import tomllib
from lexverse.config import ConfigError
from lexverse.models.task.config import UserTaskSpec


def load_user_task(path: Path) -> UserTaskSpec:
    path = path.expanduser().resolve()
    if path.is_dir():
        path /= "task.toml"
    if path.name != "task.toml":
        raise ConfigError("User tasks require task.toml; old task formats are unsupported")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        _reject_credentials(data)
        task = UserTaskSpec.model_validate(data)
        if not (path.parent / "instruction.md").read_text(encoding="utf-8").strip():
            raise ValueError("instruction.md must contain text")
        if task.verifier is not None and not (path.parent / "tests/test.sh").is_file():
            raise ValueError("A configured verifier requires tests/test.sh")
        task.environment.knowledge = [(source.expanduser() if source.expanduser().is_absolute() else path.parent / source.expanduser()).resolve()
                                      for source in task.environment.knowledge]
        return task
    except (ValueError, OSError) as exc:
        raise ConfigError(f"Invalid user task: {exc}") from exc


def _reject_credentials(value):
    if isinstance(value, dict):
        if any(str(key).lower() in {"api_key", "openai_api_key", "token", "hf_token", "authorization", "password", "headers"} for key in value):
            raise ConfigError("Credentials must be stored in host secrets, not task configuration")
        for item in value.values():
            _reject_credentials(item)
    elif isinstance(value, list):
        for item in value:
            _reject_credentials(item)


SUPPORTED_AGENTS = frozenset({"vanilla", "external"})

from pathlib import Path
import yaml
from lexverse.config import ConfigError
from lexverse.models.job.config import JobConfig


def read_job_config(path: Path) -> dict:
    path = path.expanduser().resolve()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Job configuration must be a mapping")
        return data
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigError(f"Invalid Job configuration {path}: {exc}") from exc


def load_job_config(path: Path) -> JobConfig:
    path = path.expanduser().resolve()
    try:
        return JobConfig.model_validate(read_job_config(path)).resolve(path.parent)
    except ValueError as exc:
        raise ConfigError(f"Invalid Job configuration {path}: {exc}") from exc


def write_job_config(config: JobConfig, path: Path) -> None:
    """Never overwrite an existing experiment, including through a symlink."""
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise ConfigError(f"Job configuration already exists: {path}")
    config.resolve(Path.cwd())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config.model_dump(mode="json", exclude_none=True), stream,
                       sort_keys=False, allow_unicode=True)

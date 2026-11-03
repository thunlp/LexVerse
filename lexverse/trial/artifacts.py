import fnmatch
import hashlib
from pathlib import Path
import shutil
import stat
import tempfile

from lexverse.models.task.config import ArtifactSpec


def artifact_path(root: Path, destination: str) -> Path:
    path = root / destination
    for parent in (root, *(root / part for part in Path(destination).parents if part != Path("."))):
        if parent.is_symlink():
            raise ValueError(f"Artifact destination contains a symlink: {parent}")
    return path


def clean_copy(source: Path, destination: Path, exclude=(), relative=""):
    mode = source.lstat().st_mode
    if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        raise ValueError(f"Unsupported artifact entry: {source.name}")
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        destination.chmod(0o755)
        for item in source.iterdir():
            name = f"{relative}/{item.name}" if relative else item.name
            if not any(fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(item.name, pattern) for pattern in exclude):
                clean_copy(item, destination / item.name, exclude, name)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o644)


async def collect(environment, declarations, root: Path):
    import shlex
    root.mkdir(parents=True, exist_ok=True)
    records, files, missing = [], [], []
    for value in declarations:
        item = ArtifactSpec.model_validate(value)
        selected = environment.for_service(item.service)
        source = shlex.quote(item.source)
        check = await selected.exec(f"if [ -L {source} ]; then echo link; elif [ -f {source} ]; then echo file; "
                                    f"elif [ -d {source} ]; then echo directory; else echo missing; fi")
        if check.return_code:
            raise RuntimeError(f"Cannot inspect artifact: {item.source}")
        kind = (check.stdout or "").strip()
        record = {"source": item.source, "destination": item.destination, "service": item.service,
                  "status": "missing" if kind == "missing" else "collected", "kind": kind}
        if kind == "missing":
            if item.required:
                missing.append(item.source)
            records.append(record)
            continue
        if kind not in {"file", "directory"}:
            raise ValueError(f"Unsafe artifact source: {item.source}")
        destination = artifact_path(root, item.destination)
        with tempfile.TemporaryDirectory(prefix="artifact-") as temporary:
            staged = Path(temporary) / "output"
            if kind == "directory":
                await selected.download_dir(item.source, staged)
            else:
                await selected.download_file(item.source, staged)
            if destination.exists() or destination.is_symlink():
                if destination.is_symlink() or destination.is_file():
                    destination.unlink()
                else:
                    shutil.rmtree(destination)
            clean_copy(staged, destination, item.exclude)
        records.append(record)
        for path in ([destination] if destination.is_file() else sorted(destination.rglob("*"))):
            if path.is_file():
                files.append({"name": path.relative_to(root).as_posix(), "path": str(path),
                              "size_bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "format": path.suffix.lstrip(".")})
    return records, files, missing


async def restore(environment, records, root: Path):
    for record in records:
        if record["status"] != "collected":
            continue
        selected = environment.for_service(record["service"])
        item = ArtifactSpec.model_validate({key: record[key] for key in ("source", "destination", "service")})
        source = artifact_path(root, item.destination)
        with tempfile.TemporaryDirectory(prefix="verify-artifact-") as temporary:
            staged = Path(temporary) / "output"
            clean_copy(source, staged)
            if staged.is_dir():
                await selected.upload_dir(staged, record["source"])
            else:
                await selected.upload_file(staged, record["source"])

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
import json
from lexverse.config import ConfigError


class UpstreamError(Exception):
    """Upstream source could not be prepared."""


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patchset_hash(patches: list[Path]) -> str:
    """Stable short hash over patch filenames + contents."""
    h = hashlib.sha256()
    for p in sorted(patches):
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


def run_identity(manifest: dict) -> dict:
    from lexverse.tasks.bundle import TaskBundle

    identity = {
        "config_hash": manifest["config_hash"],
        "bundle_hash": (TaskBundle.model_validate(manifest["bundle"]).content_hash()
                        if "bundle" in manifest else manifest.get("bundle_hash", manifest.get("prepared_bundle_hash"))),
        "provenance": manifest.get("provenance"),
    }
    if manifest.get("snapshot_version") in (1.0, 2) and "snapshot_hashes" not in manifest:
        identity.update(config=manifest.get("config"), selected_tasks=manifest.get("selected_tasks"),
                        data_sources=manifest.get("data_sources"))
    if "capability_identity" in manifest:
        identity["capability_identity"] = manifest["capability_identity"]
    return identity


def source_files(plugin) -> list[Path]:
    package = Path(__file__).resolve().parents[1]
    adapter = plugin.catalog_path.parent.resolve()
    common = [p for p in package.rglob("*.py") if "benchmarks" not in p.relative_to(package).parts]
    common += list((package / "benchmarks").glob("*.py"))
    local = [p for p in adapter.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".patch"}]
    return sorted(set(common + local))


def upstream_files(plugin, *, offline: bool) -> tuple[Path, list[Path]]:
    root = plugin.upstream.ensure_patched(plugin.catalog_path.parent / "patches", offline=offline)
    return root, sorted(root.rglob("*.py"))


def source_provenance(plugin, *, offline: bool = True, config=None) -> dict:
    package = Path(__file__).resolve().parents[1]
    files = [{
        "path": str(path.relative_to(package.parent)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    } for path in source_files(plugin)]
    patches = sorted((plugin.catalog_path.parent / "patches").glob("*.patch"))
    upstream_root, code = upstream_files(plugin, offline=offline)

    return {
        "upstream": {"repo_url": plugin.upstream.repo_url, "commit": plugin.upstream.commit},
        "patches": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in patches],
        "patchset_hash": _patchset_hash(patches),
        "adapter_version": plugin.adapter_version,
        "files": files,
        "source_hash": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "upstream_code": [{
            "path": str(path.relative_to(upstream_root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        } for path in code],
        "evaluation_inputs": [{
            "name": name, "path": str(path.resolve()),
            "sha256": file_hash(path),
        } for name, path in sorted(plugin.evaluation_inputs(config).items())] if config else [],
    }


@dataclass
class UpstreamSource:
    """One pinned upstream benchmark source.

    Layout under ``<root>``::

        <name>/<commit>/original/             # complete pristine clone
        <name>/<commit>/patched/<patchhash>/  # patched runtime copy
        .locks/<name>-<short_commit>.lock      # file lock for clone + patch

    Set `patches_dir` (or call `apply_patches`) to make `cache_dir` resolve
    to the patched work copy; otherwise `cache_dir` is the pristine clone.
    """

    name: str
    repo_url: str
    commit: str
    cache_root: Path | None = None  # default: .lexverse/upstream
    patches_dir: Path | None = None  # set to expose a patched work copy

    @property
    def cache_dir(self) -> Path:
        """Return the prepared patched tree, or the pristine clone."""
        if self.patches_dir is not None:
            patches = sorted(self.patches_dir.glob("*.patch"))
            if patches:
                work = self.pristine_dir.parent / "patched" / _patchset_hash(patches)
                if (work / ".lexverse-patched").exists():
                    return work
        return self.pristine_dir

    @property
    def pristine_dir(self) -> Path:
        """The never-modified clone."""
        return self._root / self.name / self.commit / "original"

    def ensure_patched(self, patches_dir: Path, *, offline: bool = False) -> Path:
        """Prepare patches and return the tree callers must use."""
        self.patches_dir = patches_dir
        return self.ensure(offline=offline)

    def ensure(self, offline: bool = False) -> Path:
        """Prepare the source and configured patches, honoring offline mode."""
        with self._lock():
            self._migrate_legacy_cache()
            cache = self.pristine_dir
            marker = cache / ".lexverse-ready"
            if not marker.exists():
                if offline:
                    raise UpstreamError(
                        f"upstream '{self.name}' not cached and offline=True "
                        f"(expected at {cache})"
                    )
                self._clone_and_checkout(cache)
                marker.write_text(f"{self.commit}\n")
            self._verify_commit(cache)

            if self.patches_dir is not None:
                return self._materialise_patched(cache)
            return cache

    def apply_patches(self, patches_dir: Path) -> Path:
        """Apply sorted patches in a reusable tree keyed by their content hash."""
        self.patches_dir = patches_dir
        return self.ensure()

    def _migrate_legacy_cache(self) -> None:
        """Reuse flat-layout caches under the existing cross-process lock."""
        legacy = self._root / f"{self.name}-{self._short}"
        if legacy.exists():
            self._verify_commit(legacy)
            if self.pristine_dir.exists():
                raise UpstreamError(f"both legacy and original caches exist: {legacy}, {self.pristine_dir}")
            self.pristine_dir.parent.mkdir(parents=True, exist_ok=True)
            legacy.rename(self.pristine_dir)
        if not (self.pristine_dir / ".lexverse-ready").exists():
            return
        for old in sorted(self._root.glob(f"{legacy.name}-w*")):
            work = self.pristine_dir.parent / "patched" / old.name[len(legacy.name) + 2:]
            if work.exists():
                raise UpstreamError(f"both legacy and patched caches exist: {old}, {work}")
            work.parent.mkdir(parents=True, exist_ok=True)
            old.rename(work)
            for name in self._work_copy_excludes():
                path = work / name
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()

    def _work_copy_excludes(self) -> set[str]:
        """Original downloads stay complete; runtime copies omit duplicate history."""
        return {".git"} | {
            "lexeval": {"model_output"},
            "lawbench": {"predictions"},
        }.get(self.name, set())

    def _materialise_patched(self, pristine: Path) -> Path:
        patches = sorted(self.patches_dir.glob("*.patch"))  # type: ignore[union-attr]
        if not patches:
            return pristine
        work = pristine.parent / "patched" / _patchset_hash(patches)
        if (work / ".lexverse-patched").exists():
            return work

        # Rebuild from pristine; a partial work copy is disposable.
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(pristine, work, ignore=lambda directory, names: (
            self._work_copy_excludes().intersection(names) if Path(directory) == pristine else set()
        ))
        try:
            for patch in patches:
                self._git_apply(work, patch)
        except UpstreamError:
            shutil.rmtree(work, ignore_errors=True)
            raise
        (work / ".lexverse-patched").write_text(
            f"patchset={_patchset_hash(patches)}\n"
            + "".join(f"{p.name}\n" for p in patches)
        )
        return work

    def _lock(self):
        """Cross-process lock around clone + patch operations."""
        from filelock import FileLock

        path = self._lock_path
        path.parent.mkdir(parents=True, exist_ok=True)
        return FileLock(str(path))

    def _clone_and_checkout(self, cache: Path) -> None:
        if cache.exists():
            shutil.rmtree(cache)
        cache.parent.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                ["git", "clone", "--filter=blob:none", self.repo_url, str(cache)],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "checkout", self.commit],
                cwd=str(cache),
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
            raise UpstreamError(
                f"failed to prepare upstream '{self.name}' "
                f"({self.repo_url} @ {self.commit}):\n{stderr}"
            ) from exc

    def _verify_commit(self, cache: Path) -> None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(cache),
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise UpstreamError(
                f"cache at {cache} is not a git repo (corrupted?)"
            ) from exc

        head = result.stdout.decode().strip()
        if not head.startswith(self.commit) and not self.commit.startswith(head):
            raise UpstreamError(
                f"upstream '{self.name}' cache commit {head} != expected {self.commit}"
            )

    def _git_apply(self, work: Path, patch: Path) -> None:
        patch_abs = patch.resolve()
        for flag in (["--check"], []):
            try:
                subprocess.run(
                    ["git", "apply", "--ignore-whitespace", *flag, str(patch_abs)],
                    cwd=str(work),
                    env={**os.environ, "GIT_CEILING_DIRECTORIES": str(work.resolve().parent)},
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
                raise UpstreamError(
                    f"failed to apply patch {patch.name}:\n{stderr}"
                ) from exc

    @property
    def _root(self) -> Path:
        if self.cache_root is not None:
            return self.cache_root
        return Path(os.environ.get("LEXVERSE_UPSTREAM_CACHE",
                                    str(Path.cwd() / ".lexverse" / "upstream")))

    @property
    def _short(self) -> str:
        return self.commit[:12]

    @property
    def _lock_path(self) -> Path:
        return self._root / ".locks" / f"{self.name}-{self._short}.lock"


def prepare_run_sources(run_dir: Path, manifest: dict, plugin, *, resume: bool,
                        offline: bool = True, config=None) -> str:
    prepared_source = manifest.get("provenance") or {}
    if prepared_source.get("upstream", {}).get("commit") != plugin.upstream.commit:
        raise ConfigError("source drift or missing provenance; start a new `lexverse benchmark run --config CONFIG`")
    current = source_provenance(plugin, offline=offline, config=config)
    code_fields = {"files", "source_hash"}
    recorded_conditions = {k: v for k, v in prepared_source.items() if k not in code_fields}
    current_conditions = {k: v for k, v in current.items() if k not in code_fields}
    if recorded_conditions != current_conditions or (not resume and prepared_source != current):
        raise ConfigError("source drift or missing provenance; start a new `lexverse benchmark run --config CONFIG`")
    identity = run_identity(manifest)
    old_path = run_dir / "manifest.json"
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        if not resume:
            raise ConfigError("run directory already has a manifest; enable resume or use a new directory")
        if run_identity(old) != identity:
            raise ConfigError("resume source/config/task mismatch; use a new run directory")
    elif (run_dir / "trials").exists() and any((run_dir / "trials").iterdir()):
        raise ConfigError("existing trials have no source manifest; use a new run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        from lexverse.runtime.results import atomic_write_json

        history_path = run_dir / "resume_history.json"
        history = json.loads(history_path.read_text()) if history_path.exists() else []
        previous = history[-1]["code"] if history else prepared_source
        before = {entry["path"]: entry["sha256"] for entry in previous.get("files", [])}
        after = {entry["path"]: entry["sha256"] for entry in current.get("files", [])}
        changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
        history.append({"resumed_at": datetime.now(timezone.utc).isoformat(),
                        "changed_files": changed,
                        "code": {key: current[key] for key in code_fields}})
        atomic_write_json(history_path, history)
        if changed:
            logging.getLogger("lexverse.run").warning(
                "[resume] local code changed; continuing with original run identity: %s", ", ".join(changed))
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

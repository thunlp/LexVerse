"""Prepare pinned upstream sources without mutating pristine clones.

Each commit has an immutable clone and patch-hash-addressed working copies
under ``~/.lexverse/upstream``. A cross-process lock protects clone and patch
materialization, and incomplete working copies can be rebuilt from pristine.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class UpstreamError(Exception):
    """Upstream source could not be prepared."""


@dataclass
class UpstreamSource:
    """One pinned upstream benchmark source.

    Layout under ``<root>``::

        <name>-<short_commit>/                 # pristine clone (read-only)
        <name>-<short_commit>-w<patchhash>/    # patched work copy (disposable)
        .locks/<name>-<short_commit>.lock      # file lock for clone + patch

    Set `patches_dir` (or call `apply_patches`) to make `cache_dir` resolve
    to the patched work copy; otherwise `cache_dir` is the pristine clone.
    """

    name: str
    repo_url: str
    commit: str
    cache_root: Path | None = None  # default: ~/.lexverse/upstream
    patches_dir: Path | None = None  # set to expose a patched work copy

    # -- paths -------------------------------------------------------------

    @property
    def _root(self) -> Path:
        if self.cache_root is not None:
            return self.cache_root
        return Path(os.environ.get("LEXVERSE_UPSTREAM_CACHE",
                                    str(Path.home() / ".lexverse" / "upstream")))

    @property
    def _short(self) -> str:
        return self.commit[:12]

    @property
    def pristine_dir(self) -> Path:
        """The never-modified clone."""
        return self._root / f"{self.name}-{self._short}"

    @property
    def cache_dir(self) -> Path:
        """The directory callers should read from.

        Returns the patched work copy when `patches_dir` is set and the
        copy exists, otherwise the pristine clone.
        """
        if self.patches_dir is not None:
            patches = sorted(self.patches_dir.glob("*.patch"))
            if patches:
                work = self._root / f"{self.name}-{self._short}-w{_patchset_hash(patches)}"
                if (work / ".lexverse-patched").exists():
                    return work
        return self.pristine_dir

    @property
    def _lock_path(self) -> Path:
        return self._root / ".locks" / f"{self.name}-{self._short}.lock"

    # -- public API --------------------------------------------------------

    def ensure(self, offline: bool = False) -> Path:
        """Return a usable source tree (patched if `patches_dir` is set).

        Clones the pristine copy on cache miss, applies patches when a
        patches_dir is configured, and returns whichever directory callers
        should read from.
        """
        with self._lock():
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
        """Materialise a patched work copy; return its path.

        All *.patch files under `patches_dir` are applied, in filename
        order. The work copy is keyed by a hash of the patchset contents, so
        editing a patch produces a new directory and never mutates the old
        one. Idempotent across runs.
        """
        self.patches_dir = patches_dir
        with self._lock():
            # ensure pristine exists without recursing into the patched path
            cache = self.pristine_dir
            if not (cache / ".lexverse-ready").exists():
                self._clone_and_checkout(cache)
                (cache / ".lexverse-ready").write_text(f"{self.commit}\n")
            self._verify_commit(cache)
            return self._materialise_patched(cache)

    def ensure_patched(self, patches_dir: Path, *, offline: bool = False) -> Path:
        """Prepare and return the patched tree while honoring offline mode.

        Callers must use the returned path: the pristine and patched trees are
        deliberately separate. This method prevents the common ordering bug of
        saving ``ensure()``'s pristine path before applying patches.
        """
        self.patches_dir = patches_dir
        return self.ensure(offline=offline)

    def _materialise_patched(self, pristine: Path) -> Path:
        patches = sorted(self.patches_dir.glob("*.patch"))  # type: ignore[union-attr]
        if not patches:
            return pristine
        work = self._root / f"{self.name}-{self._short}-w{_patchset_hash(patches)}"
        if (work / ".lexverse-patched").exists():
            return work

        # Rebuild from pristine; a partial work copy is disposable.
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(pristine, work)
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

    # kept for callers that only need the file list
    def patch_filenames(self, patches_dir: Path) -> list[str]:
        if not patches_dir.exists():
            return []
        return [p.name for p in sorted(patches_dir.glob("*.patch"))]

    # -- internals ---------------------------------------------------------

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
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
                raise UpstreamError(
                    f"failed to apply patch {patch.name}:\n{stderr}"
                ) from exc


def _patchset_hash(patches: list[Path]) -> str:
    """Stable short hash over patch filenames + contents."""
    h = hashlib.sha256()
    for p in sorted(patches):
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


def _network_ok() -> bool:
    """Whether a clone would be permitted. Offline CI sets this to '0'."""
    return os.environ.get("LEXVERSE_UPSTREAM_OFFLINE", "0") not in ("1", "true", "yes")


LEXEVAL = UpstreamSource(
    name="lexeval",
    repo_url="https://github.com/CSHaitao/LexEval",
    commit="044c695f62894deef41ad9a30797e1da0404945c",
)

J1BENCH = UpstreamSource(
    name="j1bench",
    repo_url="https://github.com/FudanDISC/J1Bench",
    commit="61a6ea5515e11229c13353e73b0629e39dcc6076",
)

LAWBENCH = UpstreamSource(
    name="lawbench",
    repo_url="https://github.com/open-compass/LawBench",
    commit="e30981bb3ff54c41571f222e0b23e92d27375388",
)

"""Internal types shared by loaders, indexing and providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol


@dataclass(frozen=True)
class RecordContext:
    collection: str
    source: str
    relative_file: str
    ordinal: int
    locator: dict[str, Any]
    object_key: str | None = None


@dataclass(frozen=True)
class ProjectionResult:
    source_key: str
    key_fields: dict[str, Any] = field(default_factory=dict)
    search_text: str = ""
    filter_fields: dict[str, Any] = field(default_factory=dict)
    asset_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoadedRecord:
    collection: str
    source: str
    source_key: str
    raw_record: Any
    locator: dict[str, Any]
    key_fields: dict[str, Any]
    search_text: str
    filter_fields: dict[str, Any]
    asset_paths: tuple[str, ...] = ()
    relative_file: str = ""
    ordinal: int = 0


Projection = Callable[[Any, RecordContext], ProjectionResult]


class Loader(Protocol):
    collection: str
    source: str

    def iter_records(self, data_root: Path) -> Iterable[LoadedRecord]:
        ...

    def describe(self) -> dict[str, Any]:
        ...

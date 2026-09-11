"""Streaming loaders for the heterogeneous files under LexVerse/data."""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Iterator

from pydantic import ValidationError

from .errors import InvalidArgumentError
from .types import LoadedRecord, ProjectionResult, RecordContext, Projection
from .utils import safe_path


class _TextBuffer:
    """Incremental UTF-8 text reader that also reports byte offsets."""

    def __init__(self, path: Path, *, chunk_size: int = 1024 * 1024):
        self.path = path
        self.chunk_size = chunk_size
        self.handle = path.open("r", encoding="utf-8", newline="")
        self.buffer = ""
        self.position = 0
        self.base_byte = 0
        self.eof = False
        self.decoder = json.JSONDecoder()

    def __enter__(self) -> "_TextBuffer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.handle.close()

    def fill(self) -> bool:
        if self.eof:
            return False
        chunk = self.handle.read(self.chunk_size)
        if chunk == "":
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def ensure(self) -> bool:
        while self.position >= len(self.buffer) and not self.eof:
            self.fill()
        return self.position < len(self.buffer)

    def skip_whitespace(self) -> None:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or self.eof:
                return
            self.fill()

    def byte_offset(self, char_position: int) -> int:
        return self.base_byte + len(self.buffer[:char_position].encode("utf-8"))

    def maybe_compact(self, *, force: bool = False) -> None:
        if self.position == 0:
            return
        if not force and self.position < 1024 * 1024 and self.position < len(self.buffer) // 2:
            return
        consumed = self.buffer[: self.position]
        self.base_byte += len(consumed.encode("utf-8"))
        self.buffer = self.buffer[self.position :]
        self.position = 0

    def raw_decode(self) -> tuple[Any, int, int]:
        while True:
            if not self.ensure():
                raise ValueError(f"JSON 文件为空或缺少值: {self.path}")
            start_char = self.position
            try:
                value, end_char = self.decoder.raw_decode(self.buffer, self.position)
            except JSONDecodeError:
                if self.eof:
                    raise
                self.fill()
                continue
            start_byte = self.byte_offset(start_char)
            end_byte = self.byte_offset(end_char)
            # Advance before yielding so the enclosing array/object parser
            # sees the delimiter after the decoded value.
            self.position = end_char
            return value, start_byte, end_byte


def iter_json_array(path: Path) -> Iterator[tuple[Any, int, int]]:
    """Yield top-level array values with byte start/end offsets."""

    with _TextBuffer(path) as reader:
        reader.skip_whitespace()
        if not reader.ensure() or reader.buffer[reader.position] != "[":
            raise ValueError(f"顶层不是 JSON 数组: {path}")
        reader.position += 1
        while True:
            reader.skip_whitespace()
            if not reader.ensure():
                raise ValueError(f"JSON 数组未闭合: {path}")
            if reader.buffer[reader.position] == "]":
                reader.position += 1
                return
            value, start, end = reader.raw_decode()
            yield value, start, end
            reader.skip_whitespace()
            if not reader.ensure():
                raise ValueError(f"JSON 数组缺少结束符: {path}")
            delimiter = reader.buffer[reader.position]
            if delimiter == ",":
                reader.position += 1
                reader.maybe_compact()
                continue
            if delimiter == "]":
                reader.position += 1
                return
            raise ValueError(f"JSON 数组元素之间缺少逗号: {path}")


def iter_json_object_items(path: Path) -> Iterator[tuple[str, Any, int, int]]:
    """Yield top-level object key/value pairs with value byte offsets."""

    with _TextBuffer(path) as reader:
        reader.skip_whitespace()
        if not reader.ensure() or reader.buffer[reader.position] != "{":
            raise ValueError(f"顶层不是 JSON 对象: {path}")
        reader.position += 1
        while True:
            reader.skip_whitespace()
            if not reader.ensure():
                raise ValueError(f"JSON 对象未闭合: {path}")
            if reader.buffer[reader.position] == "}":
                reader.position += 1
                return
            key, _, _ = reader.raw_decode()
            if not isinstance(key, str):
                raise ValueError(f"JSON 对象 key 不是字符串: {path}")
            reader.skip_whitespace()
            if not reader.ensure() or reader.buffer[reader.position] != ":":
                raise ValueError(f"JSON 对象缺少冒号: {path}")
            reader.position += 1
            reader.skip_whitespace()
            value, start, end = reader.raw_decode()
            yield key, value, start, end
            reader.skip_whitespace()
            if not reader.ensure():
                raise ValueError(f"JSON 对象缺少结束符: {path}")
            delimiter = reader.buffer[reader.position]
            if delimiter == ",":
                reader.position += 1
                reader.maybe_compact()
                continue
            if delimiter == "}":
                reader.position += 1
                return
            raise ValueError(f"JSON 对象成员之间缺少逗号: {path}")


def _relative(data_root: Path, path: Path) -> str:
    return path.relative_to(data_root).as_posix()


def _matching_files(
    data_root: Path, pattern: str, *, suffixes: set[str]
) -> list[Path]:
    """Resolve matched files through the data-root allowlist before opening them."""

    files: list[Path] = []
    for candidate in sorted(data_root.glob(pattern), key=lambda path: path.as_posix()):
        relative_file = _relative(data_root, candidate)
        path = safe_path(data_root, relative_file, allow_suffixes=suffixes)
        if path.is_file():
            files.append(path)
    return files


def _loaded(
    *,
    record: Any,
    context: RecordContext,
    projection: Projection,
) -> LoadedRecord:
    try:
        projected: ProjectionResult = projection(record, context)
    except ValidationError as exc:
        raise InvalidArgumentError(
            "数据记录无法映射到索引 schema",
            details={
                "source": context.source,
                "relative_file": context.relative_file,
                "ordinal": context.ordinal,
                "validation_errors": [
                    {"location": list(error["loc"]), "type": error["type"]}
                    for error in exc.errors(include_input=False)
                ],
            },
        ) from exc
    return LoadedRecord(
        collection=context.collection,
        source=context.source,
        source_key=projected.source_key,
        raw_record=record,
        locator=context.locator,
        key_fields=projected.key_fields,
        search_text=projected.search_text,
        filter_fields=projected.filter_fields,
        asset_paths=projected.asset_paths,
        relative_file=context.relative_file,
        ordinal=context.ordinal,
    )


class JsonlLoader:
    def __init__(
        self,
        collection: str,
        source: str,
        pattern: str,
        projection: Projection,
    ):
        self.collection = collection
        self.source = source
        self.pattern = pattern
        self.projection = projection

    def _files(self, data_root: Path) -> list[Path]:
        return _matching_files(data_root, self.pattern, suffixes={".jsonl"})

    def iter_records(self, data_root: Path) -> Iterator[LoadedRecord]:
        for path in self._files(data_root):
            relative_file = _relative(data_root, path)
            with path.open("rb") as handle:
                byte_offset = 0
                for line_number, raw_line in enumerate(handle, start=1):
                    line_start = byte_offset
                    byte_offset += len(raw_line)
                    content = raw_line.strip()
                    if not content:
                        continue
                    record = json.loads(content)
                    context = RecordContext(
                        collection=self.collection,
                        source=self.source,
                        relative_file=relative_file,
                        ordinal=line_number,
                        locator={
                            "kind": "bytes",
                            "relative_file": relative_file,
                            "byte_start": line_start,
                            "byte_end": byte_offset,
                        },
                    )
                    yield _loaded(record=record, context=context, projection=self.projection)

    def describe(self) -> dict[str, Any]:
        return {
            "loader": "jsonl",
            "pattern": self.pattern,
            "record_granularity": "one non-empty line",
        }


class JsonArrayLoader:
    def __init__(
        self,
        collection: str,
        source: str,
        pattern: str,
        projection: Projection,
    ):
        self.collection = collection
        self.source = source
        self.pattern = pattern
        self.projection = projection

    def _files(self, data_root: Path) -> list[Path]:
        return _matching_files(data_root, self.pattern, suffixes={".json"})

    def iter_records(self, data_root: Path) -> Iterator[LoadedRecord]:
        for path in self._files(data_root):
            relative_file = _relative(data_root, path)
            for ordinal, (record, start, end) in enumerate(iter_json_array(path), start=1):
                context = RecordContext(
                    collection=self.collection,
                    source=self.source,
                    relative_file=relative_file,
                    ordinal=ordinal,
                    locator={
                        "kind": "bytes",
                        "relative_file": relative_file,
                        "byte_start": start,
                        "byte_end": end,
                    },
                )
                yield _loaded(record=record, context=context, projection=self.projection)

    def describe(self) -> dict[str, Any]:
        return {
            "loader": "json_array",
            "pattern": self.pattern,
            "record_granularity": "one top-level array item",
            "streaming": True,
        }


class KeyedObjectLoader:
    def __init__(
        self,
        collection: str,
        source: str,
        pattern: str,
        projection: Projection,
    ):
        self.collection = collection
        self.source = source
        self.pattern = pattern
        self.projection = projection

    def _files(self, data_root: Path) -> list[Path]:
        return _matching_files(data_root, self.pattern, suffixes={".json"})

    def iter_records(self, data_root: Path) -> Iterator[LoadedRecord]:
        for path in self._files(data_root):
            relative_file = _relative(data_root, path)
            for ordinal, (object_key, record, start, end) in enumerate(
                iter_json_object_items(path),
                start=1,
            ):
                context = RecordContext(
                    collection=self.collection,
                    source=self.source,
                    relative_file=relative_file,
                    ordinal=ordinal,
                    object_key=object_key,
                    locator={
                        "kind": "bytes",
                        "relative_file": relative_file,
                        "byte_start": start,
                        "byte_end": end,
                    },
                )
                yield _loaded(record=record, context=context, projection=self.projection)

    def describe(self) -> dict[str, Any]:
        return {
            "loader": "keyed_object",
            "pattern": self.pattern,
            "record_granularity": "one top-level object member",
            "streaming": True,
        }


class FilePerRecordLoader:
    def __init__(
        self,
        collection: str,
        source: str,
        pattern: str,
        projection: Projection,
    ):
        self.collection = collection
        self.source = source
        self.pattern = pattern
        self.projection = projection

    def _files(self, data_root: Path) -> list[Path]:
        return _matching_files(data_root, self.pattern, suffixes={".json"})

    def iter_records(self, data_root: Path) -> Iterator[LoadedRecord]:
        for ordinal, path in enumerate(self._files(data_root), start=1):
            relative_file = _relative(data_root, path)
            with path.open("rb") as handle:
                raw_bytes = handle.read()
            record = json.loads(raw_bytes)
            context = RecordContext(
                collection=self.collection,
                source=self.source,
                relative_file=relative_file,
                ordinal=ordinal,
                locator={
                    "kind": "bytes",
                    "relative_file": relative_file,
                    "byte_start": 0,
                    "byte_end": len(raw_bytes),
                },
            )
            yield _loaded(record=record, context=context, projection=self.projection)

    def describe(self) -> dict[str, Any]:
        return {
            "loader": "file_per_record",
            "pattern": self.pattern,
            "record_granularity": "one file",
        }


class BatchJsonLoader(JsonlLoader):
    """Compatibility name for batch-shaped sources already normalized to JSONL."""

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description["loader"] = "batch_json"
        return description


class TemplateAssetLoader(JsonlLoader):
    """Load template metadata whose projection registers related file assets."""

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description.update(
            {
                "loader": "template_asset",
                "record_granularity": "one metadata line with optional assets",
            }
        )
        return description

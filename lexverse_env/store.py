"""Read-only access to raw records and registered file assets."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import re
import sqlite3
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .errors import InvalidArgumentError, ReadLimitExceededError, RecordNotFoundError
from .index import read_meta
from .utils import safe_path


class RecordStore:
    def __init__(
        self,
        db: sqlite3.Connection,
        data_root: str | Path,
        *,
        max_record_bytes: int = 12 * 1024 * 1024,
        max_part_chars: int = 200_000,
        max_asset_bytes: int = 12 * 1024 * 1024,
        max_part_record_bytes: int = 256 * 1024 * 1024,
    ):
        self.db = db
        self.data_root = Path(data_root).expanduser().resolve()
        self.max_record_bytes = max_record_bytes
        self.max_part_chars = max_part_chars
        self.max_asset_bytes = max_asset_bytes
        self.max_part_record_bytes = max_part_record_bytes
        self.meta = read_meta(db)

    def _record_row(self, record_id: str) -> sqlite3.Row:
        if not isinstance(record_id, str) or not record_id:
            raise InvalidArgumentError("record_id 必须是非空字符串")
        row = self.db.execute(
            """
            SELECT id, collection, provider, source, source_key,
                   key_fields_json, locator_json
            FROM records WHERE id = ?
            """,
            (record_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"记录不存在: {record_id}")
        return row

    def _load_bytes(self, locator: dict[str, Any], *, max_bytes: int | None = None) -> Any:
        if locator.get("kind") != "bytes":
            raise InvalidArgumentError("不支持的内部记录定位类型")
        relative_file = locator.get("relative_file")
        start = locator.get("byte_start")
        end = locator.get("byte_end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise InvalidArgumentError("记录定位的字节范围无效")
        size = end - start
        limit = self.max_record_bytes if max_bytes is None else max_bytes
        if limit is not None and size > limit:
            raise ReadLimitExceededError(
                "记录超过单次读取大小，请使用 read_record_part 读取字段片段",
                details={"max_record_bytes": limit},
            )
        path = safe_path(self.data_root, relative_file, allow_suffixes={".json", ".jsonl"})
        with path.open("rb") as handle:
            handle.seek(start)
            raw = handle.read(size)
        if len(raw) != size:
            raise RecordNotFoundError("记录定位超出原始文件范围")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RecordNotFoundError("原始记录无法解析") from exc

    def _load_raw(self, row: sqlite3.Row, *, enforce_limit: bool = True) -> Any:
        return self._load_bytes(
            json.loads(row["locator_json"]),
            max_bytes=self.max_record_bytes if enforce_limit else self.max_part_record_bytes,
        )

    def get_record(self, record_id: str) -> dict[str, Any]:
        row = self._record_row(record_id)
        envelope: dict[str, Any] = {
            "id": row["id"],
            "record": self._load_raw(row),
            "provenance": {
                "provider": row["provider"],
                "collection": row["collection"],
                "source": row["source"],
                "source_key": row["source_key"],
                "snapshot_id": self.meta.get("snapshot_id"),
            },
        }
        assets = self.db.execute(
            """
            SELECT asset_id, mime_type, byte_size
            FROM assets WHERE record_id = ? ORDER BY asset_id
            """,
            (record_id,),
        ).fetchall()
        if assets:
            envelope["assets"] = [
                {
                    "id": asset["asset_id"],
                    "mime_type": asset["mime_type"],
                    "byte_size": asset["byte_size"],
                }
                for asset in assets
            ]
        return envelope

    @staticmethod
    def _json_pointer(value: Any, pointer: str) -> Any:
        if pointer == "":
            return value
        if not isinstance(pointer, str) or not pointer.startswith("/"):
            raise InvalidArgumentError("path 必须是 JSON Pointer")
        current = value
        for part in pointer[1:].split("/"):
            if re.search(r"~(?:[^01]|$)", part):
                raise InvalidArgumentError("path 包含无效的 JSON Pointer 转义")
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict):
                if part not in current:
                    raise RecordNotFoundError(f"记录字段不存在: {pointer}")
                current = current[part]
            elif isinstance(current, list):
                if not part.isdigit() or (len(part) > 1 and part.startswith("0")):
                    raise RecordNotFoundError(f"记录字段不存在: {pointer}")
                try:
                    current = current[int(part)]
                except (ValueError, IndexError) as exc:
                    raise RecordNotFoundError(f"记录字段不存在: {pointer}") from exc
            else:
                raise RecordNotFoundError(f"记录字段不存在: {pointer}")
        return current

    def read_record_part(
        self,
        record_id: str,
        path: str,
        *,
        offset: int = 0,
        limit: int = 12_000,
    ) -> dict[str, Any]:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise InvalidArgumentError("offset 必须是非负整数")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise InvalidArgumentError("limit 必须是正整数")
        if limit > self.max_part_chars:
            raise ReadLimitExceededError(
                "字段片段超过大小限制",
                details={"max_part_chars": self.max_part_chars},
            )
        row = self._record_row(record_id)
        value = self._json_pointer(self._load_raw(row, enforce_limit=False), path)
        if isinstance(value, str):
            chunk = value[offset : offset + limit]
            next_offset = offset + len(chunk) if offset + len(chunk) < len(value) else None
            return {
                "id": record_id,
                "path": path,
                "offset": offset,
                "value": chunk,
                "next_offset": next_offset,
                "truncated": next_offset is not None,
            }
        if isinstance(value, list):
            chunk = value[offset : offset + limit]
            next_offset = offset + len(chunk) if offset + len(chunk) < len(value) else None
            return {
                "id": record_id,
                "path": path,
                "offset": offset,
                "value": chunk,
                "next_offset": next_offset,
                "truncated": next_offset is not None,
            }
        if offset:
            raise InvalidArgumentError("非字符串或数组字段只支持 offset=0")
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded) > limit:
            raise ReadLimitExceededError(
                "非字符串字段无法在当前 limit 内返回",
                details={"serialized_length": len(encoded), "limit": limit},
            )
        return {
            "id": record_id,
            "path": path,
            "offset": 0,
            "value": value,
            "next_offset": None,
            "truncated": False,
        }

    def _asset_row(self, asset_id: str) -> sqlite3.Row:
        if not isinstance(asset_id, str) or not asset_id:
            raise InvalidArgumentError("asset_id 必须是非空字符串")
        row = self.db.execute(
            """
            SELECT asset_id, record_id, relative_file, mime_type, byte_size
            FROM assets WHERE asset_id = ?
            """,
            (asset_id,),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"资产不存在: {asset_id}")
        return row

    def _asset_path(self, row: sqlite3.Row) -> Path:
        return safe_path(
            self.data_root,
            row["relative_file"],
            allow_suffixes={".docx", ".doc", ".pdf", ".txt"},
        )

    def _asset_bytes(self, row: sqlite3.Row) -> bytes:
        path = self._asset_path(row)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise RecordNotFoundError("资产文件不可读取") from exc
        if size > self.max_asset_bytes:
            raise ReadLimitExceededError(
                "资产超过单次读取大小，请先使用 metadata 或部署受控下载接口",
                details={"max_asset_bytes": self.max_asset_bytes},
            )
        try:
            with path.open("rb") as handle:
                value = handle.read(self.max_asset_bytes + 1)
        except OSError as exc:
            raise RecordNotFoundError("资产文件不可读取") from exc
        if len(value) > self.max_asset_bytes:
            raise ReadLimitExceededError(
                "资产超过单次读取大小",
                details={"max_asset_bytes": self.max_asset_bytes},
            )
        return value

    @staticmethod
    def _docx_text(raw: bytes) -> str:
        """Extract visible Word text without requiring python-docx."""

        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                document_xml = archive.read("word/document.xml")
            root = ET.fromstring(document_xml)
        except (KeyError, OSError, ET.ParseError, zipfile.BadZipFile) as exc:
            raise InvalidArgumentError("DOCX 文本提取失败") from exc

        word_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs: list[str] = []
        for paragraph in root.iter(f"{word_ns}p"):
            pieces: list[str] = []
            for child in paragraph.iter():
                if child.tag == f"{word_ns}t" and child.text:
                    pieces.append(child.text)
                elif child.tag == f"{word_ns}tab":
                    pieces.append("\t")
                elif child.tag == f"{word_ns}br":
                    pieces.append("\n")
            paragraphs.append("".join(pieces))
        return "\n".join(paragraphs)

    def get_asset(self, asset_id: str, mode: str = "metadata") -> dict[str, Any]:
        """Read a registered asset without exposing its physical path."""

        if not isinstance(mode, str) or mode not in {"metadata", "text", "binary"}:
            raise InvalidArgumentError("mode 只能是 metadata、text 或 binary")
        row = self._asset_row(asset_id)
        path = self._asset_path(row)
        suffix = path.suffix.lower()
        mime_type = row["mime_type"] or mimetypes.guess_type(path.name)[0]
        metadata: dict[str, Any] = {
            "asset_id": row["asset_id"],
            "record_id": row["record_id"],
            "filename": path.name,
            "mime_type": mime_type or "application/octet-stream",
            "byte_size": row["byte_size"],
        }
        if mode == "metadata":
            return metadata

        raw = self._asset_bytes(row)
        if mode == "binary":
            return {
                **metadata,
                "encoding": "base64",
                "data": base64.b64encode(raw).decode("ascii"),
            }

        if suffix == ".txt":
            try:
                value = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise InvalidArgumentError("TXT 资产不是有效的 UTF-8") from exc
        elif suffix == ".docx":
            value = self._docx_text(raw)
        else:
            raise InvalidArgumentError(
                "该资产类型不支持 text，请使用 metadata 或 binary"
            )
        if len(value) > self.max_part_chars:
            raise ReadLimitExceededError(
                "资产文本超过单次读取大小，请使用模板 record 的 content 或受控下载接口",
                details={"max_part_chars": self.max_part_chars},
            )
        return {**metadata, "encoding": "utf-8", "text": value}

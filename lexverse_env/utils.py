"""Small deterministic helpers used by the local data environment."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from .errors import InvalidArgumentError, InvalidCursorError, PermissionDeniedError


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def stable_hash(value: Any, length: int = 24) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def tokenize(text: Any) -> list[str]:
    """Tokenize Chinese text deterministically without a model dependency."""

    normalized = normalize_text(text)
    tokens: list[str] = []
    index = 0
    while index < len(normalized):
        char = normalized[index]
        if _is_cjk(char):
            tokens.append(char)
            if index + 1 < len(normalized) and _is_cjk(normalized[index + 1]):
                tokens.append(normalized[index : index + 2])
            index += 1
            continue
        if char.isalnum() or char == "_":
            end = index + 1
            while end < len(normalized):
                candidate = normalized[end]
                if _is_cjk(candidate) or not (candidate.isalnum() or candidate == "_"):
                    break
                end += 1
            tokens.append(normalized[index:end])
            index = end
            continue
        index += 1
    return tokens


def tokenize_for_index(text: Any, max_chars: int = 120_000) -> str:
    normalized = normalize_text(text)[:max_chars]
    return " ".join(tokenize(normalized))


def make_fts_query(query: str) -> str | None:
    tokens = list(dict.fromkeys(tokenize(query)))
    if not tokens:
        return None
    escaped = [token.replace('"', '""') for token in tokens]
    return " OR ".join(f'"{token}"' for token in escaped)


def short_value(value: Any, max_length: int = 240) -> Any:
    if isinstance(value, str):
        return value[:max_length]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [short_value(item, max_length) for item in value[:10]]
    return str(value)[:max_length]


def first_present(record: Any, names: Iterable[str]) -> Any:
    if not isinstance(record, dict):
        return None
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            return value
    return None


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return normalize_text(value)
    return ""


def flatten_text(value: Any, *, max_chars: int = 120_000) -> str:
    """Collect source text for indexing while bounding one record's size."""

    pieces: list[str] = []

    def visit(node: Any) -> None:
        if sum(len(piece) for piece in pieces) >= max_chars:
            return
        if isinstance(node, str):
            pieces.append(node)
            return
        if isinstance(node, (int, float, bool)):
            pieces.append(str(node))
            return
        if isinstance(node, dict):
            for key, child in node.items():
                if isinstance(key, str):
                    pieces.append(key)
                visit(child)
                if sum(len(piece) for piece in pieces) >= max_chars:
                    return
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                visit(child)
                if sum(len(piece) for piece in pieces) >= max_chars:
                    return

    visit(value)
    return " ".join(pieces)[:max_chars]


def encode_cursor(payload: dict[str, Any]) -> str:
    raw = canonical_json(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> dict[str, Any]:
    if not isinstance(cursor, str) or not cursor:
        raise InvalidCursorError("cursor 不是有效的字符串")
    try:
        padding = "=" * (-len(cursor) % 4)
        value = base64.urlsafe_b64decode((cursor + padding).encode("ascii"))
        payload = json.loads(value.decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("cursor 无法解析") from exc
    if not isinstance(payload, dict):
        raise InvalidCursorError("cursor 载荷无效")
    return payload


def safe_path(root: Path, relative_file: str, *, allow_suffixes: set[str] | None = None) -> Path:
    """Resolve a registered relative path without allowing traversal."""

    if not isinstance(relative_file, str) or not relative_file:
        raise PermissionDeniedError("无效的内部文件定位")
    candidate = Path(relative_file)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PermissionDeniedError("禁止访问数据根目录之外的文件")
    if allow_suffixes and candidate.suffix.lower() not in allow_suffixes:
        raise PermissionDeniedError("文件类型不在允许列表中")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise PermissionDeniedError("禁止访问数据根目录之外的文件") from exc
    return resolved


def validate_limit(value: Any, *, default: int = 10, maximum: int = 50) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentError("limit 必须是整数")
    if value < 1 or value > maximum:
        raise InvalidArgumentError(f"limit 必须在 1 到 {maximum} 之间")
    return value

"""Default source registry and source-specific search projections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .loaders import (
    BatchJsonLoader,
    FilePerRecordLoader,
    JsonArrayLoader,
    JsonlLoader,
    KeyedObjectLoader,
    TemplateAssetLoader,
)
from .types import LoadedRecord, Projection, ProjectionResult, RecordContext
from .utils import (
    first_present,
    flatten_text,
    normalize_text,
    short_value,
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {_as_text(child)}" for key, child in value.items())
    return str(value)


def _path_category(context: RecordContext) -> str | None:
    parts = context.relative_file.split("/")
    if context.collection == "legal_laws" and len(parts) >= 2:
        return parts[1]
    return None


def _year(record: Any, context: RecordContext) -> str | None:
    candidates = [
        first_present(
            record,
            [
                "effective_from",
                "effective_to",
                "发布时间",
                "published_at",
                "date",
                "判决日期",
                "裁判日期",
            ],
        ),
        context.relative_file,
    ]
    for value in candidates:
        match = re.search(r"(?:19|20)\d{2}", _as_text(value))
        if match:
            return match.group(0)
    return None


def _put_if_value(target: dict[str, Any], name: str, value: Any) -> None:
    if value not in (None, ""):
        target[name] = short_value(value)


def _source_key(record: Any, context: RecordContext) -> str:
    relative = context.relative_file
    if context.object_key is not None:
        return context.object_key

    if context.collection == "legal_laws":
        law_name = first_present(record, ["law_name", "法律名称", "name"])
        article = first_present(record, ["article", "article_number", "条文", "条号"])
        effective = first_present(record, ["effective_from", "生效日期"])
        components = [value for value in (law_name, article, effective) if value not in (None, "")]
        if components:
            return f"{_path_category(context) or 'laws'}::{':'.join(map(str, components))}"

    explicit = first_present(
        record,
        [
            "id",
            "caseID",
            "CaseId",
            "case_id",
            "pid",
            "text_id",
            "uid",
            "uniqid",
        ],
    )
    if explicit not in (None, ""):
        return str(explicit)

    if context.collection == "legal_concepts":
        explicit = first_present(record, ["entity_name", "term", "name"])
    elif context.collection == "legal_templates":
        explicit = first_present(record, ["file_path", "title", "name"])
    elif context.collection == "legal_qa":
        explicit = first_present(record, ["id", "question", "topic", "theme"])
    if explicit not in (None, ""):
        return str(explicit)
    return f"{relative}::ordinal-{context.ordinal}"


def _key_fields(record: Any, context: RecordContext) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if context.collection == "legal_laws":
        _put_if_value(fields, "law_name", first_present(record, ["law_name", "法律名称", "name"]))
        _put_if_value(
            fields,
            "article",
            first_present(record, ["article", "article_number", "条文", "条号"]),
        )
        _put_if_value(fields, "category", _path_category(context))
        _put_if_value(fields, "effective_from", first_present(record, ["effective_from", "生效日期"]))
    elif context.collection == "legal_cases":
        aliases = [
            ("case_id", ["id", "caseID", "CaseId", "case_id", "pid", "text_id", "uid"]),
            ("title", ["title", "案件名", "case_name", "name"]),
            ("case_number", ["case_number", "案号", "裁判文书案号"]),
            ("case_cause", ["case_cause", "案由", "纠纷类型", "类别"]),
            ("court", ["court", "法院", "审理法院"]),
            ("stage", ["stage", "审理程序", "审级"]),
        ]
        for name, names in aliases:
            _put_if_value(fields, name, first_present(record, names))
        _put_if_value(fields, "source", context.source)
    elif context.collection == "legal_qa":
        for name, names in [
            ("id", ["id"]),
            ("question", ["question", "topic", "opening"]),
            ("theme", ["theme", "topic_name", "category"]),
        ]:
            _put_if_value(fields, name, first_present(record, names))
        _put_if_value(fields, "source", context.source)
    elif context.collection == "legal_concepts":
        _put_if_value(fields, "term", first_present(record, ["term", "entity_name", "name"]))
        _put_if_value(fields, "description", first_present(record, ["concept", "description"]))
        _put_if_value(fields, "category", first_present(record, ["category", "domain"]))
        _put_if_value(fields, "source", context.source)
    elif context.collection == "legal_templates":
        _put_if_value(fields, "title", first_present(record, ["title", "name"]))
        _put_if_value(fields, "template_type", first_present(record, ["template_type", "类型"]))
        _put_if_value(fields, "case_type", first_present(record, ["case_type", "案由"]))
        _put_if_value(fields, "source", context.source)
    return fields


def _filter_fields(record: Any, context: RecordContext) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "collection": context.collection,
        "source": context.source,
    }
    category = _path_category(context)
    if category:
        fields["category"] = category

    for canonical, names in [
        ("dataset", ["dataset", "数据集", "来源"]),
        ("category", ["category", "类别", "分类"]),
        ("case_type", ["case_type", "案由", "纠纷类型"]),
        ("court", ["court", "法院", "审理法院"]),
        ("stage", ["stage", "审理程序", "审级"]),
    ]:
        value = first_present(record, names)
        if value not in (None, "") and canonical not in fields:
            fields[canonical] = short_value(value)
    year = _year(record, context)
    if year:
        fields["year"] = year
    return fields


def _search_text(record: Any, context: RecordContext) -> str:
    if context.collection == "legal_laws":
        values = [
            first_present(record, ["law_name", "法律名称", "name"]),
            first_present(record, ["article", "article_number", "条文", "条号"]),
            _path_category(context),
            first_present(record, ["text", "content", "正文"]),
        ]
        return " ".join(_as_text(value) for value in values if value not in (None, ""))
    if context.collection == "legal_qa":
        values = [
            first_present(record, ["question", "opening"]),
            first_present(record, ["topic", "theme", "topic_name"]),
            first_present(record, ["content", "answer"]),
        ]
        return " ".join(_as_text(value) for value in values if value not in (None, ""))
    if context.collection == "legal_concepts":
        values = [
            first_present(record, ["term", "entity_name", "name"]),
            first_present(record, ["concept", "description"]),
            first_present(record, ["facts", "attributes", "labels"]),
        ]
        return " ".join(_as_text(value) for value in values if value not in (None, ""))
    if context.collection == "legal_templates":
        values = [
            first_present(record, ["title", "name"]),
            first_present(record, ["template_type", "case_type"]),
            first_present(record, ["content", "description"]),
        ]
        return " ".join(_as_text(value) for value in values if value not in (None, ""))
    return flatten_text(record, max_chars=120_000)


def project_record(record: Any, context: RecordContext) -> ProjectionResult:
    """Build only the internal index projection; raw record stays untouched."""

    asset_paths: tuple[str, ...] = ()
    if context.collection == "legal_templates" and isinstance(record, dict):
        file_path = record.get("file_path")
        if isinstance(file_path, str) and file_path and not Path(file_path).is_absolute():
            if ".." not in Path(file_path).parts:
                asset_paths = (f"legal_templates/{context.source}/{file_path}",)

    return ProjectionResult(
        source_key=_source_key(record, context),
        key_fields=_key_fields(record, context),
        search_text=_search_text(record, context),
        filter_fields=_filter_fields(record, context),
        asset_paths=asset_paths,
    )


@dataclass(frozen=True)
class SourceDefinition:
    collection: str
    source: str
    loader_kind: str
    pattern: str
    projection: Projection = project_record

    def build_loader(self):
        args = (self.collection, self.source, self.pattern, self.projection)
        if self.loader_kind == "jsonl":
            return JsonlLoader(*args)
        if self.loader_kind == "batch_json":
            return BatchJsonLoader(*args)
        if self.loader_kind == "json_array":
            return JsonArrayLoader(*args)
        if self.loader_kind == "keyed_object":
            return KeyedObjectLoader(*args)
        if self.loader_kind == "file_per_record":
            return FilePerRecordLoader(*args)
        if self.loader_kind == "template_asset":
            return TemplateAssetLoader(*args)
        raise ValueError(f"unknown loader kind: {self.loader_kind}")

    def describe(self) -> dict[str, Any]:
        return {
            "collection": self.collection,
            "source": self.source,
            **self.build_loader().describe(),
        }


class SourceRegistry:
    def __init__(self, definitions: Iterable[SourceDefinition]):
        self._definitions = tuple(definitions)
        self._by_collection = {}
        for definition in self._definitions:
            self._by_collection.setdefault(definition.collection, []).append(definition)

    def definitions(self) -> tuple[SourceDefinition, ...]:
        return self._definitions

    def collections(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_collection))

    def for_collection(self, collection: str) -> tuple[SourceDefinition, ...]:
        return tuple(self._by_collection.get(collection, ()))

    def describe(self) -> list[dict[str, Any]]:
        return [definition.describe() for definition in self._definitions]

    @classmethod
    def default(cls) -> "SourceRegistry":
        definitions = [
            SourceDefinition("legal_laws", "local_laws", "jsonl", "legal_laws/**/*.jsonl"),
            SourceDefinition("legal_cases", "j1bench", "jsonl", "legal_cases/j1bench/**/*.jsonl"),
            SourceDefinition("legal_cases", "agentscourt", "json_array", "legal_cases/agentscourt/*.json"),
            SourceDefinition("legal_cases", "legal_world", "json_array", "legal_cases/legal_world/*.json"),
            SourceDefinition("legal_cases", "maser", "json_array", "legal_cases/maser/*.json"),
            SourceDefinition("legal_cases", "judge", "jsonl", "legal_cases/judge/*.jsonl"),
            SourceDefinition("legal_cases", "judge", "json_array", "legal_cases/judge/*.json"),
            SourceDefinition("legal_cases", "lecardv2", "file_per_record", "legal_cases/lecardv2/**/*.json"),
            SourceDefinition("legal_cases", "mslr_bench", "file_per_record", "legal_cases/mslr_bench/**/*.json"),
            SourceDefinition("legal_cases", "muser", "keyed_object", "legal_cases/muser/*.json"),
            SourceDefinition("legal_cases", "lexchain", "json_array", "legal_cases/lexchain/*.json"),
            SourceDefinition("legal_qa", "j1bench", "jsonl", "legal_qa/j1bench/**/*.jsonl"),
            SourceDefinition("legal_qa", "fadawang", "batch_json", "legal_qa/fadawang/**/*.jsonl"),
            SourceDefinition("legal_qa", "dlawbench", "jsonl", "legal_qa/dlawbench/**/*.jsonl"),
            SourceDefinition(
                "legal_concepts",
                "chinese_legal_terms",
                "jsonl",
                "legal_concepts/chinese_legal_terms/**/*.jsonl",
            ),
            SourceDefinition("legal_concepts", "ownthink", "jsonl", "legal_concepts/ownthink/**/*.jsonl"),
            SourceDefinition(
                "legal_templates",
                "china_court",
                "template_asset",
                "legal_templates/china_court/**/*.jsonl",
            ),
            SourceDefinition(
                "legal_templates",
                "supreme_court",
                "template_asset",
                "legal_templates/supreme_court/**/*.jsonl",
            ),
        ]
        return cls(definitions)


def public_collection_label(collection: str) -> str:
    return {
        "legal_laws": "法律法规",
        "legal_cases": "法律案件",
        "legal_qa": "法律问答",
        "legal_concepts": "法律概念",
        "legal_templates": "文书模板",
    }.get(collection, collection)


def collection_short_name(collection: str) -> str:
    return {
        "legal_laws": "law",
        "legal_cases": "case",
        "legal_qa": "qa",
        "legal_concepts": "concept",
        "legal_templates": "template",
    }.get(collection, collection)

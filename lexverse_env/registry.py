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

from .schema import (
    BaseRecordModel,
    LegalLawModel,
    LegalCaseModel,
    LegalQaModel,
    LegalConceptModel,
    LegalTemplateModel,
)

def project_record(record: Any, context: RecordContext) -> ProjectionResult:
    """Build internal index projection using Pydantic models; raw record stays untouched."""
    
    # 构造统一的输入数据包，注入上下文元数据
    raw_data = record if isinstance(record, dict) else {"raw": record}
    model_data = {
        **raw_data,
        "ctx_collection": context.collection,
        "ctx_source": context.source,
        "ctx_relative_file": context.relative_file,
        "ctx_ordinal": context.ordinal,
        "ctx_object_key": context.object_key,
    }

    # 根据 collection 分发到对应的 Pydantic 模型
    if context.collection == "legal_laws":
        model = LegalLawModel(**model_data)
        search_text = model.export_search_text()
    elif context.collection == "legal_cases":
        model = LegalCaseModel(**model_data)
        # 案件集合采用原版 flatten_text 兜底展平机制
        search_text = model.export_search_text() or flatten_text(record, max_chars=120_000)
    elif context.collection == "legal_qa":
        model = LegalQaModel(**model_data)
        search_text = model.export_search_text()
    elif context.collection == "legal_concepts":
        model = LegalConceptModel(**model_data)
        search_text = model.export_search_text()
    elif context.collection == "legal_templates":
        model = LegalTemplateModel(**model_data)
        search_text = model.export_search_text()
    else:
        model = BaseRecordModel(**model_data)
        search_text = flatten_text(record, max_chars=120_000)

    # 资产路径提取逻辑保持不变，依然依赖原逻辑判断
    asset_paths: tuple[str, ...] = ()
    if context.collection == "legal_templates" and isinstance(record, dict):
        file_path = record.get("file_path")
        if isinstance(file_path, str) and file_path and not Path(file_path).is_absolute():
            if ".." not in Path(file_path).parts:
                asset_paths = (f"legal_templates/{context.source}/{file_path}",)

    return ProjectionResult(
        source_key=model.export_source_key(),
        key_fields=model.export_key_fields(),
        search_text=search_text,
        filter_fields=model.export_filter_fields(),
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

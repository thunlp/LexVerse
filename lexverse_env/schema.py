"""
Pydantic schemas for LexVerse data collections.
统一实体解析，严格分离输出视图。
"""

import re
from typing import Annotated, Optional

from pydantic import AliasChoices, BaseModel, BeforeValidator, Field

from .utils import flatten_text


def _text_or_empty(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return flatten_text(value)
    return str(value)


def _optional_text(value):
    return None if value is None else _text_or_empty(value)


TextValue = Annotated[str, BeforeValidator(_text_or_empty)]
OptionalTextValue = Annotated[Optional[str], BeforeValidator(_optional_text)]


def _clean_dict(data: dict) -> dict:
    """辅助函数：过滤掉 None 或空字符串的值，对应原代码的 _put_if_value 行为"""
    return {k: str(v)[:240] if isinstance(v, str) else v for k, v in data.items() if v not in (None, "")}


class BaseRecordModel(BaseModel):
    """所有 Collection 共用的基础数据模型，吸收上下文并提供公共导出逻辑"""
    
    model_config = {"extra": "allow"}
    
    # 注入物理文件上下文参数（exclude=True 保证不污染主数据字典）
    ctx_collection: str = Field(default="", exclude=True)
    ctx_source: str = Field(default="", exclude=True)
    ctx_relative_file: str = Field(default="", exclude=True)
    ctx_ordinal: int = Field(default=0, exclude=True)
    ctx_object_key: Optional[str] = Field(default=None, exclude=True)
    
    # 共用字段别名映射
    dataset: OptionalTextValue = Field(default=None, validation_alias=AliasChoices("dataset", "数据集", "来源"))
    category: OptionalTextValue = Field(default=None, validation_alias=AliasChoices("category", "类别", "分类"))
    
    # 统一接收可能包含年份信息的杂乱字段
    raw_date: OptionalTextValue = Field(
        default=None, 
        validation_alias=AliasChoices("effective_from", "effective_to", "发布时间", "published_at", "date", "判决日期", "裁判日期", "生效日期"),
        exclude=True
    )

    def get_computed_category(self) -> Optional[str]:
        """复刻原 _path_category()，优先从文件路径推断"""
        parts = self.ctx_relative_file.split("/")
        if self.ctx_collection == "legal_laws" and len(parts) >= 2:
            return parts[1]
        return self.category

    def get_computed_year(self) -> Optional[str]:
        """复刻原 _year()，支持基于字段正则和物理文件名的兜底推断"""
        for candidate in [self.raw_date, self.ctx_relative_file]:
            if candidate:
                match = re.search(r"(?:19|20)\d{2}", str(candidate))
                if match:
                    return match.group(0)
        return None

    def export_source_key(self) -> str:
        """兜底的 Source Key 生成逻辑"""
        if self.ctx_object_key:
            return self.ctx_object_key
        return f"{self.ctx_relative_file}::ordinal-{self.ctx_ordinal}"


class LegalLawModel(BaseRecordModel):
    """法律法规集合 (legal_laws)"""
    
    law_name: TextValue = Field(default="", validation_alias=AliasChoices("law_name", "法律名称", "name"))
    article: TextValue = Field(default="", validation_alias=AliasChoices("article", "article_number", "条文", "条号"))
    effective_from: TextValue = Field(default="", validation_alias=AliasChoices("effective_from", "生效日期"))
    text: TextValue = Field(default="", validation_alias=AliasChoices("text", "content", "正文"))

    def export_source_key(self) -> str:
        if self.ctx_object_key:
            return self.ctx_object_key
        components = [v for v in (self.law_name, self.article, self.effective_from) if v]
        if components:
            return f"{self.get_computed_category() or'laws'}::{':'.join(components)}"
        return super().export_source_key()

    def export_key_fields(self) -> dict:
        return _clean_dict({
            "law_name": self.law_name,
            "article": self.article,
            "category": self.get_computed_category(),
            "effective_from": self.effective_from
        })

    def export_filter_fields(self) -> dict:
        return _clean_dict({
            "collection": self.ctx_collection,
            "source": self.ctx_source,
            "category": self.get_computed_category(),
            "year": self.get_computed_year()
        })

    def export_search_text(self) -> Optional[str]:
        values = [self.law_name, self.article, self.get_computed_category(), self.text]
        return " ".join(str(v) for v in values if v not in (None, ""))


class LegalCaseModel(BaseRecordModel):
    """法律案件集合 (legal_cases)"""
    
    case_id: TextValue = Field(default="", validation_alias=AliasChoices("id", "caseID", "CaseId", "case_id", "pid", "text_id", "uid", "uniqid"))
    title: TextValue = Field(default="", validation_alias=AliasChoices("title", "案件名", "case_name", "name"))
    case_number: TextValue = Field(default="", validation_alias=AliasChoices("case_number", "案号", "裁判文书案号"))
    case_cause: TextValue = Field(default="", validation_alias=AliasChoices("case_cause", "案由", "纠纷类型", "类别", "case_type"))
    court: TextValue = Field(default="", validation_alias=AliasChoices("court", "法院", "审理法院"))
    stage: TextValue = Field(default="", validation_alias=AliasChoices("stage", "审理程序", "审级"))

    def export_source_key(self) -> str:
        if self.ctx_object_key:
            return self.ctx_object_key
        if self.case_id:
            return self.case_id
        return super().export_source_key()

    def export_key_fields(self) -> dict:
        return _clean_dict({
            "case_id": self.case_id,
            "title": self.title,
            "case_number": self.case_number,
            "case_cause": self.case_cause,
            "court": self.court,
            "stage": self.stage,
            "source": self.ctx_source
        })

    def export_filter_fields(self) -> dict:
        return _clean_dict({
            "collection": self.ctx_collection,
            "source": self.ctx_source,
            "category": self.get_computed_category(),
            "dataset": self.dataset,
            "case_type": self.case_cause,  # 映射为下游要求的 case_type
            "court": self.court,
            "stage": self.stage,
            "year": self.get_computed_year()
        })

    def export_search_text(self) -> Optional[str]:
        # 返回 None，通知外层继续使用原代码中的 flatten_text 兜底机制展平字典
        return None


class LegalQaModel(BaseRecordModel):
    """法律问答集合 (legal_qa)"""
    
    qa_id: TextValue = Field(default="", validation_alias=AliasChoices("id", "question", "topic", "theme"))
    question: TextValue = Field(default="", validation_alias=AliasChoices("question", "topic", "opening"))
    theme: TextValue = Field(default="", validation_alias=AliasChoices("theme", "topic_name", "category"))
    content: TextValue = Field(default="", validation_alias=AliasChoices("content", "answer"))

    def export_source_key(self) -> str:
        if self.ctx_object_key:
            return self.ctx_object_key
        if self.qa_id:
            return self.qa_id
        return super().export_source_key()

    def export_key_fields(self) -> dict:
        return _clean_dict({
            "id": self.qa_id,
            "question": self.question,
            "theme": self.theme,
            "source": self.ctx_source
        })

    def export_filter_fields(self) -> dict:
        return _clean_dict({
            "collection": self.ctx_collection,
            "source": self.ctx_source,
            "dataset": self.dataset,
            "category": self.get_computed_category(),
            "case_type": self.category, # 问答集合无明确案由映射，根据原代码可能留空或使用全局分类
            "year": self.get_computed_year()
        })

    def export_search_text(self) -> Optional[str]:
        values = [self.question, self.theme, self.content]
        return " ".join(str(v) for v in values if v not in (None, ""))


class LegalConceptModel(BaseRecordModel):
    """法律概念集合 (legal_concepts)"""
    
    term: TextValue = Field(default="", validation_alias=AliasChoices("term", "entity_name", "name"))
    description: TextValue = Field(default="", validation_alias=AliasChoices("concept", "description"))
    facts: TextValue = Field(default="", validation_alias=AliasChoices("facts", "attributes", "labels"))
    concept_domain: TextValue = Field(default="", validation_alias=AliasChoices("category", "domain"))

    def export_source_key(self) -> str:
        if self.ctx_object_key:
            return self.ctx_object_key
        if self.term:
            return self.term
        return super().export_source_key()

    def export_key_fields(self) -> dict:
        return _clean_dict({
            "term": self.term,
            "description": self.description,
            "category": self.concept_domain,
            "source": self.ctx_source
        })

    def export_filter_fields(self) -> dict:
        return _clean_dict({
            "collection": self.ctx_collection,
            "source": self.ctx_source,
            "dataset": self.dataset,
            "category": self.get_computed_category() or self.concept_domain,
            "year": self.get_computed_year()
        })

    def export_search_text(self) -> Optional[str]:
        values = [self.term, self.description, self.facts]
        return " ".join(str(v) for v in values if v not in (None, ""))


class LegalTemplateModel(BaseRecordModel):
    """文书模板集合 (legal_templates)"""
    
    title: TextValue = Field(default="", validation_alias=AliasChoices("file_path", "title", "name"))
    template_type: TextValue = Field(default="", validation_alias=AliasChoices("template_type", "类型"))
    case_type: TextValue = Field(default="", validation_alias=AliasChoices("case_type", "案由"))
    content: TextValue = Field(default="", validation_alias=AliasChoices("content", "description"))

    def export_source_key(self) -> str:
        if self.ctx_object_key:
            return self.ctx_object_key
        if self.title:
            return self.title
        return super().export_source_key()

    def export_key_fields(self) -> dict:
        return _clean_dict({
            "title": self.title,
            "template_type": self.template_type,
            "case_type": self.case_type,
            "source": self.ctx_source
        })

    def export_filter_fields(self) -> dict:
        return _clean_dict({
            "collection": self.ctx_collection,
            "source": self.ctx_source,
            "category": self.get_computed_category(),
            "case_type": self.case_type,
            "year": self.get_computed_year()
        })

    def export_search_text(self) -> Optional[str]:
        values = [self.title, self.template_type, self.case_type, self.content]
        return " ".join(str(v) for v in values if v not in (None, ""))

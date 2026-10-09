from langchain_core.tools import tool
from pydantic import BaseModel, Field, create_model
from typing import Literal
from lexverse.tasks.user import KnowledgeFilters


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=3000)
    source_ids: list[str] | None = None
    filters: KnowledgeFilters | None = None
    top_k: int = Field(default=8, ge=1, le=20)


class FetchArgs(BaseModel):
    citation_id: str
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=4000, ge=1, le=8000)


def knowledge_tools(retriever, selected: list[str]):
    sources = tuple(retriever.indexes)
    source_type = list[Literal[sources]] if sources else list[str]
    source_field = Field(default=None) if sources else Field(default=None, max_length=0)
    search_args = create_model("TaskSearchArgs", __base__=SearchArgs,
                               source_ids=(source_type | None, source_field))

    @tool(args_schema=search_args)
    def search_knowledge(query: str, source_ids: list[str] | None = None, top_k: int = 8,
                         filters: KnowledgeFilters | None = None) -> list[dict]:
        """Search allowed knowledge sources and return excerpts with citation IDs."""
        return retriever.search(query, top_k=top_k, source_ids=source_ids,
                                filters=filters.model_dump(exclude_none=True) if filters is not None else None)

    @tool(args_schema=FetchArgs)
    def fetch_knowledge(citation_id: str, offset: int = 0, limit: int = 4000) -> dict:
        """Read a frozen knowledge excerpt by citation ID, with pagination."""
        return retriever.fetch(citation_id, offset=offset, limit=limit)

    return [operation for operation in (search_knowledge, fetch_knowledge) if operation.name in selected]


@tool
def ask_user(question: str, facts_summary: str | None = None) -> str:
    """Ask the user for information or confirmation of a factual summary."""
    raise RuntimeError("ask_user must be intercepted by human-in-the-loop middleware")

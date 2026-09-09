"""LexVerse's read-only, agent-facing data environment."""

from .env import CorpusEnv
from .errors import LexVerseError, error_response
from .index import IndexBuilder
from .providers import LocalProvider, PkulawMcpProvider, ProviderRouter
from .query import SearchRequest
from .registry import SourceDefinition, SourceRegistry

__all__ = [
    "CorpusEnv",
    "IndexBuilder",
    "LexVerseError",
    "LocalProvider",
    "PkulawMcpProvider",
    "ProviderRouter",
    "SearchRequest",
    "SourceDefinition",
    "SourceRegistry",
    "error_response",
]

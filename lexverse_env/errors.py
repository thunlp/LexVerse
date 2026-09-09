"""Errors exposed by the read-only LexVerse data environment."""

from __future__ import annotations

import re


_LOCAL_PATH_RE = re.compile(
    r"(?<![\w])/(?:Users|private|tmp|var|home|opt|Volumes)/[^\s\u3001\uff0c\uff1b;]+"
)
_SENSITIVE_DETAIL_KEY_RE = re.compile(
    r"(?:token|secret|password|authorization|api[_-]?key)", re.IGNORECASE
)


def _public_message(message: str) -> str:
    """Avoid leaking host paths when an expected error crosses the boundary."""

    return _LOCAL_PATH_RE.sub("<path>", message)


def _public_details(value):
    if isinstance(value, str):
        return _public_message(value)
    if isinstance(value, dict):
        return {
            str(key): (
                "<redacted>"
                if _SENSITIVE_DETAIL_KEY_RE.search(str(key))
                else _public_details(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_public_details(item) for item in value]
    return value


class LexVerseError(Exception):
    """Base class for expected environment errors."""

    code = "internal_error"
    retryable = False

    def __init__(self, message: str, *, details: dict | None = None):
        self.message = _public_message(message)
        super().__init__(self.message)
        self.details = _public_details(details or {})

    def to_dict(self) -> dict:
        return {
            "error": {
                "code": self.code,
                "message": _public_message(self.message),
                "retryable": self.retryable,
                "details": _public_details(self.details),
            }
        }


class InvalidArgumentError(LexVerseError):
    code = "invalid_argument"


class UnknownCollectionError(LexVerseError):
    code = "unknown_collection"


class UnknownProviderError(LexVerseError):
    code = "unknown_provider"


class InvalidCursorError(LexVerseError):
    code = "invalid_cursor"


class RecordNotFoundError(LexVerseError):
    code = "not_found"


class ReadLimitExceededError(LexVerseError):
    code = "read_limit_exceeded"


class ProviderTimeoutError(LexVerseError):
    code = "provider_timeout"
    retryable = True


class ProviderUnavailableError(LexVerseError):
    code = "provider_unavailable"
    retryable = True


class PermissionDeniedError(LexVerseError):
    code = "permission_denied"


class EnvironmentNotReadyError(LexVerseError):
    code = "environment_not_ready"


def error_response(exc: Exception) -> dict:
    """Convert an expected or unexpected exception to a safe JSON response."""

    if isinstance(exc, LexVerseError):
        return exc.to_dict()
    return {
        "error": {
            "code": "internal_error",
            "message": "环境内部错误",
            "retryable": False,
            "details": {},
        }
    }

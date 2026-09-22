from __future__ import annotations

from collections.abc import Callable

from .base import InteractionPolicy
from .direct_response import DirectResponsePolicy


PolicyFactory = Callable[[], InteractionPolicy]
_POLICIES: dict[str, PolicyFactory] = {"direct_response": DirectResponsePolicy}


def register_policy(name: str, factory: PolicyFactory) -> None:
    if name in _POLICIES:
        raise ValueError(f"interaction policy {name!r} is already registered")
    _POLICIES[name] = factory


def create_policy(name: str) -> InteractionPolicy:
    try:
        return _POLICIES[name]()
    except KeyError as exc:
        available = ", ".join(sorted(_POLICIES)) or "<none>"
        raise KeyError(f"unknown interaction policy {name!r}; available: {available}") from exc


def available_policies() -> list[str]:
    return sorted(_POLICIES)

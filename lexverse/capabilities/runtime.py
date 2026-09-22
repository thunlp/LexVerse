from __future__ import annotations

from .base import CapabilitySelection


class CapabilityRuntime:
    """Optional capability seam. Disabled selections are a valid no-op."""

    async def prepare(self, selection: CapabilitySelection) -> dict:
        if not selection.enabled:
            return {"enabled": False, "calls": []}
        raise NotImplementedError(
            "LexVerse optional capabilities are not implemented; the current "
            "runtime accepts only enabled=false"
        )

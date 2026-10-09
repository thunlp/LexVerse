from __future__ import annotations

import asyncio


class TrialError(Exception):
    pass


class TrialTimeoutError(TrialError, asyncio.TimeoutError):
    pass


class TrialProcessError(TrialError):
    def __init__(self, returncode: int, stderr_tail: str = "") -> None:
        message = f"trial subprocess exited with code {returncode}"
        lines = [line.strip() for line in stderr_tail.splitlines() if line.strip()]
        if lines:
            message += f": {lines[-1]}"
        super().__init__(message)
        self.returncode = returncode
        self.stderr_tail = stderr_tail


class ArtifactMissingError(TrialError):
    def __init__(self, expected: str) -> None:
        super().__init__(f"expected artifact not found: {expected}")
        self.expected = expected


class EvaluationError(TrialError):
    pass


class ProviderError(TrialError):
    pass

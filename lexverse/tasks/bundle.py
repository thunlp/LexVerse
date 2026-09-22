from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel

from .schema import LexVerseTask


class TaskBundle(BaseModel):
    schema_version: int = 1
    benchmark: str
    source_version: str
    tasks: list[LexVerseTask]

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json")
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()

# Copyright 2025-2026 Alex Shaw and the Harbor Framework Team
# Portions of this file are derived from Harbor (https://github.com/laude-institute/harbor, v0.20.0).
# Modified by the LexVerse Contributors for LexVerse package integration and legal capability support.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Content models for multimodal ATIF trajectories.

Added in ATIF-v1.6 to support multimodal content (images) in trajectories.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ImageSource(BaseModel):
    """Image source specification for images stored as files or at remote URLs."""

    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = Field(
        default=...,
        description="MIME type of the image",
    )
    path: str = Field(
        default=...,
        description=(
            "Location of the image. Can be a relative or absolute file path, or a URL."
        ),
    )

    model_config = {"extra": "forbid"}


class ContentPart(BaseModel):
    """A single content part within a multimodal message.

    Used when a message or observation contains mixed content types (text and images).
    For text-only content, a plain string can still be used instead of a ContentPart array.
    """

    type: Literal["text", "image"] = Field(
        default=...,
        description="The type of content",
    )
    text: str | None = Field(
        default=None,
        description="Text content. Required when type='text'.",
    )
    source: ImageSource | None = Field(
        default=None,
        description="Image source (file reference). Required when type='image'.",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_content_type(self) -> "ContentPart":
        """Validate that the correct fields are present for each content type."""
        if self.type == "text":
            if self.text is None:
                raise ValueError("'text' field is required when type='text'")
            if self.source is not None:
                raise ValueError("'source' field is not allowed when type='text'")
        elif self.type == "image":
            if self.source is None:
                raise ValueError("'source' field is required when type='image'")
            if self.text is not None:
                raise ValueError("'text' field is not allowed when type='image'")
        return self

"""Platform_Spec data model.

Encodes per-platform creative constraints (character limits, allowed creative
types, forbidden symbols) used by Creative_Generator and Keyword_Embedder to
enforce Requirements 2.2, 2.3, 2.4 and 5.3.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from creative_agent.models.enums import Creative_Type, Target_Platform


class Platform_Spec(BaseModel):
    """Per-platform copy specification.

    Attributes:
        platform: The Target_Platform this spec applies to.
        char_limits: Maximum character length permitted for each Creative_Type.
        allowed_creative_types: Creative_Type values supported on this platform.
        forbidden_symbols: Literal symbol sequences that must not appear in copy.
        notes: Optional free-form notes for human reviewers.
    """

    model_config = ConfigDict(frozen=True)

    platform: Target_Platform
    char_limits: dict[Creative_Type, int]
    allowed_creative_types: list[Creative_Type]
    forbidden_symbols: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

    def char_limit(self, creative_type: Creative_Type) -> int:
        """Return the maximum character length for the given creative type.

        Raises:
            KeyError: If the creative type is not configured for this platform.
        """
        try:
            return self.char_limits[creative_type]
        except KeyError as exc:
            raise KeyError(
                f"Platform {self.platform.value} has no char_limit configured "
                f"for creative_type {creative_type.value}"
            ) from exc

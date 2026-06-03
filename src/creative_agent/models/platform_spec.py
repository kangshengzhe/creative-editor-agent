"""Platform_Spec data model.

Encodes per-platform creative constraints (character limits, allowed creative
types, forbidden symbols) used by Creative_Generator and Keyword_Embedder to
enforce Requirements 2.2, 2.3, 2.4 and 5.3.

Display-width limits (Requirement 4.10): when ``use_display_width`` is True the
values in ``char_limits`` are interpreted as Display_Unit counts rather than raw
character counts. A Display_Unit is the visual width unit used by ad platforms
(ASCII counts as 1, CJK counts as 2). The Display_Width_Calculator is the single
source of truth for converting text to Display_Unit counts; this model only
stores the limit values and the flag describing how they should be interpreted.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from creative_agent.models.enums import Creative_Type, Target_Platform


class Platform_Spec(BaseModel):
    """Per-platform copy specification.

    Attributes:
        platform: The Target_Platform this spec applies to.
        char_limits: Maximum length permitted for each Creative_Type. These are
            interpreted as raw character counts when ``use_display_width`` is
            False, and as Display_Unit counts (CJK = 2, ASCII = 1) when
            ``use_display_width`` is True.
        allowed_creative_types: Creative_Type values supported on this platform.
        forbidden_symbols: Literal symbol sequences that must not appear in copy.
        use_display_width: When True, the values in ``char_limits`` (including
            headline and description limits) are interpreted as Display_Units and
            must be enforced via the Display_Width_Calculator rather than
            ``len()``. Defaults to False for backward compatibility, in which
            case limits are plain character counts.
        notes: Optional free-form notes for human reviewers.
    """

    model_config = ConfigDict(frozen=True)

    platform: Target_Platform
    char_limits: dict[Creative_Type, int]
    allowed_creative_types: list[Creative_Type]
    forbidden_symbols: list[str] = Field(default_factory=list)
    use_display_width: bool = False
    notes: Optional[str] = None

    def char_limit(self, creative_type: Creative_Type) -> int:
        """Return the maximum length for the given creative type.

        The returned value is interpreted as a raw character count when
        ``use_display_width`` is False, and as a Display_Unit count (see
        Requirement 4.10) when ``use_display_width`` is True. In the
        display-width case the value should be compared against the output of the
        Display_Width_Calculator rather than ``len()``.

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

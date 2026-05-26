"""Platform_Spec configuration loader.

Loads per-platform creative specs from JSON files bundled in
``src/creative_agent/config/platform_specs/``. Results are cached in-process so
repeated lookups in a request lifecycle are O(1).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from creative_agent.models.enums import Target_Platform
from creative_agent.models.platform_spec import Platform_Spec

_PLATFORM_SPECS_DIR: Path = Path(__file__).resolve().parent / "platform_specs"

# Map enum value to the on-disk filename (kept explicit for safety against
# future enum renames).
_PLATFORM_FILES: dict[Target_Platform, str] = {
    Target_Platform.GOOGLE_ADS: "google_ads.json",
    Target_Platform.FACEBOOK_ADS: "facebook_ads.json",
    Target_Platform.TIKTOK_ADS: "tiktok_ads.json",
}


@lru_cache(maxsize=None)
def load_platform_spec(platform: Target_Platform) -> Platform_Spec:
    """Load the Platform_Spec for the given target platform.

    Results are cached for the lifetime of the process via ``functools.lru_cache``.

    Args:
        platform: The target platform whose spec should be loaded.

    Returns:
        A ``Platform_Spec`` populated from the bundled JSON config.

    Raises:
        FileNotFoundError: If no config file is registered for the platform or
            the file is missing on disk.
        ValueError: If the JSON contents fail Pydantic validation.
    """
    filename = _PLATFORM_FILES.get(platform)
    if filename is None:
        raise FileNotFoundError(
            f"No platform spec config registered for platform {platform!r}"
        )

    config_path = _PLATFORM_SPECS_DIR / filename
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Platform spec config file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    return Platform_Spec.model_validate(raw)


def clear_cache() -> None:
    """Clear the platform spec cache. Primarily useful in tests."""
    load_platform_spec.cache_clear()

"""Enumerations for Creative Editor Agent.

All enums inherit from ``str`` and ``Enum`` so that they serialize to plain
JSON strings (consumed unchanged by both Pydantic v2 ``model_dump_json`` and
manual ``json.dumps`` calls). Values mirror the contracts in
``design.md`` → *Data Models* exactly.
"""

from __future__ import annotations

from enum import Enum


class Target_Platform(str, Enum):
    """Supported ad platforms (requirement 1.3, design Data Models)."""

    GOOGLE_ADS = "GOOGLE_ADS"
    FACEBOOK_ADS = "FACEBOOK_ADS"
    TIKTOK_ADS = "TIKTOK_ADS"


class Target_Market(str, Enum):
    """Supported target markets (requirement 1.4)."""

    PH = "PH"
    TH = "TH"
    RU = "RU"
    EN_GLOBAL = "EN_GLOBAL"


class Target_Language(str, Enum):
    """Supported target languages for localization (requirements 4.1-4.4)."""

    EN = "en"
    FIL = "fil"
    TH = "th"
    RU = "ru"


class Creative_Type(str, Enum):
    """Creative copy types (requirement 1.5, design Data Models)."""

    HEADLINE = "HEADLINE"
    DESCRIPTION = "DESCRIPTION"
    CTA = "CTA"
    LONG_COPY = "LONG_COPY"


class Compliance_Severity(str, Enum):
    """Severity of a compliance violation (design § Compliance_Report)."""

    BLOCK = "BLOCK"
    WARN = "WARN"
    INFO = "INFO"


class ViolationCategory(str, Enum):
    """Categories of compliance violations (design § Compliance_Report)."""

    MISLEADING = "MISLEADING"
    GAMBLING = "GAMBLING"
    MEDICAL_PROMISE = "MEDICAL_PROMISE"
    DISCRIMINATION = "DISCRIMINATION"
    FALSE_URGENCY = "FALSE_URGENCY"
    SENSITIVE_EVENT = "SENSITIVE_EVENT"
    EMPTY_COPY = "EMPTY_COPY"
    EXAGGERATION = "EXAGGERATION"


__all__ = [
    "Target_Platform",
    "Target_Market",
    "Target_Language",
    "Creative_Type",
    "Compliance_Severity",
    "ViolationCategory",
]

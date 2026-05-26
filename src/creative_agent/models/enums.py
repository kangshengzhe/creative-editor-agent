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
    """Supported target markets — covers all Coco regions."""

    # Asia
    PH = "PH"          # Philippines
    TH = "TH"          # Thailand
    VN = "VN"          # Vietnam
    ID = "ID"          # Indonesia
    MY = "MY"          # Malaysia
    SG = "SG"          # Singapore
    KH = "KH"          # Cambodia
    HK = "HK"          # Hong Kong
    TW = "TW"          # Taiwan
    JP = "JP"          # Japan
    KR = "KR"          # South Korea
    IN = "IN"          # India
    PK = "PK"          # Pakistan
    KZ = "KZ"          # Kazakhstan

    # Middle East
    SA = "SA"          # Saudi Arabia
    AE = "AE"          # UAE
    QA = "QA"          # Qatar
    BH = "BH"          # Bahrain
    KW = "KW"          # Kuwait
    OM = "OM"          # Oman

    # Africa
    EG = "EG"          # Egypt
    GH = "GH"          # Ghana
    KE = "KE"          # Kenya
    NG = "NG"          # Nigeria
    TZ = "TZ"          # Tanzania
    UG = "UG"          # Uganda

    # Americas
    BR = "BR"          # Brazil
    MX = "MX"          # Mexico
    CO = "CO"          # Colombia
    CL = "CL"          # Chile
    PE = "PE"          # Peru
    US = "US"          # United States
    BO = "BO"          # Bolivia
    GT = "GT"          # Guatemala
    PY = "PY"          # Paraguay
    CR = "CR"          # Costa Rica
    DO = "DO"          # Dominican Republic
    EC = "EC"          # Ecuador

    # Europe & Other
    RU = "RU"          # Russia
    TR = "TR"          # Turkey
    GB = "GB"          # United Kingdom
    EU = "EU"          # European Union
    AU = "AU"          # Australia

    # Global fallback
    EN_GLOBAL = "EN_GLOBAL"


class Target_Language(str, Enum):
    """Supported target languages for localization."""

    EN = "en"           # English
    FIL = "fil"         # Filipino (Tagalog)
    TH = "th"           # Thai
    VI = "vi"           # Vietnamese
    ID = "id"           # Indonesian (Bahasa Indonesia)
    MS = "ms"           # Malay
    KM = "km"           # Khmer (Cambodian)
    ZH_HK = "zh-HK"    # Chinese (Hong Kong Traditional)
    ZH_TW = "zh-TW"    # Chinese (Taiwan Traditional)
    JA = "ja"           # Japanese
    KO = "ko"           # Korean
    HI = "hi"           # Hindi
    UR = "ur"           # Urdu
    KK = "kk"           # Kazakh
    AR = "ar"           # Arabic
    PT_BR = "pt-BR"     # Portuguese (Brazil)
    ES = "es"           # Spanish
    RU = "ru"           # Russian
    TR = "tr"           # Turkish
    SW = "sw"           # Swahili


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

# src/enums.py
# +---------------------------------------------------------------------------+
# |                            ENUMS                                       |
# +---------------------------------------------------------------------------+

# Enumerated Types that act as constants for specific fields.

# Python Libraries
from enum import IntEnum, StrEnum


class RequestType(StrEnum):
    FEATURE_REQ = "feature_request"
    PRODUCT_ISSUE = "product_issue"
    BUG = "bug"
    INVALID = "invalid"


class Status(StrEnum):
    REPLIED = "Replied"
    ESCALATED = "Escalated"


class RagStatus(IntEnum):
    SUCCESS = 2
    PARTIAL = 1
    FAIL = 0


class Company(StrEnum):
    HACKERRANK = "HackerRank"
    CLAUDE = "Claude"
    VISA = "Visa"
    NONE = "None"


class Risk(StrEnum):
    LOW = "low"
    MED = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Urgency(StrEnum):
    NORMAL = "normal"

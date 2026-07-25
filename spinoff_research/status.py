"""
Extraction/review status model shared by field_extraction_runs and
field_values. Explicit states only — never a blank string or zero to
represent "unknown."
"""
from enum import Enum


class FieldStatus(str, Enum):
    PENDING = "pending"
    EXTRACTED_HIGH_CONFIDENCE = "extracted_high_confidence"
    EXTRACTED_UNCERTAIN = "extracted_uncertain"
    NOT_FOUND = "not_found"
    CONFLICTING_SOURCES = "conflicting_sources"
    NOT_YET_DETERMINABLE = "not_yet_determinable"
    REQUIRES_MANUAL_REVIEW = "requires_manual_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    MANUALLY_ENTERED = "manually_entered"


class ReviewAction(str, Enum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"
    MARK_NOT_FOUND = "mark_not_found"

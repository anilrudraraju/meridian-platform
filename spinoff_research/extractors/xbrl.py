"""
XBRL-backed extractors: spinoff_shares_outstanding, spinoff_debt,
last_year_sales, parent_shares_outstanding. Thin wrappers around
xbrl_service.resolve_field_for_transaction() — that function already does
all the real work (candidate resolution, period anchoring, conflict
detection); this module only translates its XbrlFieldResult into the
common ExtractedFieldValue shape for persistence.
"""
from typing import Optional

from spinoff_research.extraction import ExtractedFieldValue, SourceCitation
from spinoff_research.status import FieldStatus
from spinoff_research.xbrl_service import (
    fetch_company_facts,
    resolve_field_for_transaction,
    XbrlServiceError,
    _ANNUAL_FIELDS,
)

_XBRL_STATUS_MAP = {
    "extracted_high_confidence": FieldStatus.EXTRACTED_HIGH_CONFIDENCE,
    "extracted_uncertain": FieldStatus.EXTRACTED_UNCERTAIN,
    "not_found": FieldStatus.NOT_FOUND,
    "conflicting_sources": FieldStatus.CONFLICTING_SOURCES,
}

# XBRL-derived confidence, distinct from the qualitative status: multiple
# agreeing candidates and a tight period match are more trustworthy than a
# single unverified candidate, even when both land on the same status.
_CONFIDENCE_BY_STATUS = {
    "extracted_high_confidence": 0.9,
    "extracted_uncertain": 0.55,
    "conflicting_sources": 0.3,
    "not_found": None,
}


def extract_xbrl_field(
    field_key: str,
    cik: str,
    distribution_date: Optional[str] = None,
    announcement_date: Optional[str] = None,
) -> ExtractedFieldValue:
    """
    field_key must be one of xbrl_service._FIELD_TO_XBRL_CONCEPTS's keys
    (spinoff_shares_outstanding, spinoff_debt, last_year_sales, or —
    passing the parent's CIK — parent_shares_outstanding). Snapshot fields
    need distribution_date; the annual-duration field (last_year_sales)
    needs announcement_date — see resolve_field_for_transaction's own
    docstring for why these can't share one anchor.
    """
    try:
        facts = fetch_company_facts(cik)
    except XbrlServiceError as e:
        return ExtractedFieldValue(
            field_key=field_key, extraction_method="xbrl", status=FieldStatus.NOT_FOUND,
            sources=[SourceCitation(reasoning_summary=f"Could not fetch XBRL company facts: {e}")],
        )

    result = resolve_field_for_transaction(
        facts, field_key,
        distribution_date=distribution_date, announcement_date=announcement_date,
    )

    status = _XBRL_STATUS_MAP.get(result.status, FieldStatus.NOT_FOUND)
    confidence = _CONFIDENCE_BY_STATUS.get(result.status)

    sources = []
    if result.selected:
        c = result.selected
        sources.append(SourceCitation(
            supporting_excerpt=f"{c.concept} = {c.value} {c.unit}, period {c.period_start or ''}..{c.period_end}",
            reasoning_summary=f"{result.selection_reason} (accession {c.accession_number}, form {c.form})",
        ))
    elif result.rejected_candidates:
        # conflicting_sources: no selection, but the conflict itself is the evidence
        sources.append(SourceCitation(
            reasoning_summary=result.selection_reason,
            supporting_excerpt=result.conflict_note,
        ))

    sel = result.selected
    return ExtractedFieldValue(
        field_key=field_key,
        extraction_method="xbrl",
        status=status,
        raw_value=str(sel.value) if sel else None,
        numeric_value=sel.value if sel else None,
        unit=sel.unit if sel else None,
        reporting_period=f"{sel.period_start or ''}..{sel.period_end}" if sel else None,
        as_of_date=sel.period_end if sel else None,
        value_basis="historical",
        confidence=confidence,
        sources=sources,
    )

"""
Filing-metadata extractors: form_10_availability, parent/spinoff company
name and ticker. All sourced from sec_service.py's CIK resolution and
filing discovery — no document content is ever read for these, they're
pure SEC API metadata lookups.
"""
from typing import Optional

from spinoff_research.extraction import ExtractedFieldValue, SourceCitation
from spinoff_research.sec_service import discover_filings
from spinoff_research.status import FieldStatus


def extract_form_10_availability(spinoff_cik: str) -> ExtractedFieldValue:
    """True if any 10-12B/10-12G (Form 10) filing exists for the spinco CIK."""
    filings = discover_filings(
        spinoff_cik, form_types=["10-12B", "10-12B/A", "10-12G", "10-12G/A"]
    )
    found = len(filings) > 0
    citation = None
    if found:
        f = sorted(filings, key=lambda x: x.filing_date)[0]
        citation = SourceCitation(
            supporting_excerpt=f"{f.form_type} filed {f.filing_date}",
            reasoning_summary=f"Found via EDGAR submissions API filtered to Form 10 form types for CIK {spinoff_cik}.",
        )
    return ExtractedFieldValue(
        field_key="form_10_availability",
        extraction_method="filing_metadata",
        status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE if found else FieldStatus.NOT_FOUND,
        raw_value=str(found),
        normalized_value=str(found),
        confidence=1.0 if found else None,
        sources=[citation] if citation else [],
    )


def extract_company_identity(ticker: str, resolved: Optional[dict], field_key_prefix: str) -> list:
    """
    Returns [name_result, ticker_result] for either 'parent' or 'spinoff'
    (field_key_prefix). resolved is sec_service.resolve_cik()'s return
    value — {"cik": ..., "name": ...} or None if the ticker wasn't found.
    Both fields share one CIK lookup, so this is a single function rather
    than two independent extractors that would each re-resolve the ticker.
    """
    name_key = f"{field_key_prefix}_company_name" if field_key_prefix == "parent" else "spinoff_company_name"
    ticker_key = f"{field_key_prefix}_ticker"

    if resolved is None:
        not_found = FieldStatus.NOT_FOUND
        return [
            ExtractedFieldValue(field_key=name_key, extraction_method="filing_metadata", status=not_found),
            ExtractedFieldValue(field_key=ticker_key, extraction_method="filing_metadata", status=not_found),
        ]

    citation = SourceCitation(
        reasoning_summary=f"Resolved from SEC company_tickers.json for ticker '{ticker}'.",
    )
    return [
        ExtractedFieldValue(
            field_key=name_key, extraction_method="filing_metadata",
            status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE,
            raw_value=resolved["name"], normalized_value=resolved["name"],
            confidence=1.0, sources=[citation],
        ),
        ExtractedFieldValue(
            field_key=ticker_key, extraction_method="filing_metadata",
            status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE,
            raw_value=ticker.upper(), normalized_value=ticker.upper(),
            confidence=1.0, sources=[citation],
        ),
    ]

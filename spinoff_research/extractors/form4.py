"""
Form 4-backed extractors: insider_buying_within_3mo,
cluster_insider_buying_within_3mo, insider_buyer_count. Wraps
sec_service.discover_filings() (to find Form 4s in the window) +
form4_parser.parse_form4_xml()/summarize_insider_buying() (already proven
against all 3 pilots, matched Rich's own manual counts exactly).

All three fields share ONE aggregation pass — computing them separately
would mean parsing the same Form 4s three times.
"""
from datetime import datetime, timedelta
from typing import List

from spinoff_research.extraction import ExtractedFieldValue, SourceCitation
from spinoff_research.sec_service import discover_filings, _paced_get, SecServiceError
from spinoff_research.form4_parser import parse_form4_xml, summarize_insider_buying, Form4ParseError
from spinoff_research.status import FieldStatus


def extract_insider_buying_fields(
    spinoff_cik: str,
    distribution_date: str,
    window_days: int = 90,
    cluster_threshold: int = 3,
) -> List[ExtractedFieldValue]:
    """
    Returns [insider_buying_within_3mo, cluster_insider_buying_within_3mo,
    insider_buyer_count] as a single 3-element list. window_days=90
    matches the field dictionary's "3 months" definition;
    cluster_threshold=3 is the same configurable default
    form4_parser.summarize_insider_buying() uses — both are explicit
    parameters here, not buried constants, since the field dictionary
    flags the cluster threshold as a judgment call to confirm with Rich.
    """
    window_start = distribution_date
    window_end = (
        datetime.strptime(distribution_date, "%Y-%m-%d") + timedelta(days=window_days)
    ).strftime("%Y-%m-%d")

    filings = discover_filings(
        spinoff_cik, form_types=["4"], filed_after=window_start, filed_before=window_end,
    )

    all_transactions = []
    parse_failures = 0
    for f in filings:
        try:
            resp = _paced_get(f.raw_document_url)
            all_transactions.extend(parse_form4_xml(resp.content, document_id=f.accession_number_dashed))
        except (SecServiceError, Form4ParseError):
            parse_failures += 1

    summary = summarize_insider_buying(
        all_transactions, window_start=window_start, window_end=window_end,
        cluster_threshold=cluster_threshold,
    )

    evidence_note = (
        f"{len(filings)} Form 4 filings examined in [{window_start}, {window_end}]"
        + (f", {parse_failures} failed to parse" if parse_failures else "")
    )
    buyer_names = sorted({tx.reporting_owner_name for tx in summary.purchase_transactions})
    citation = SourceCitation(
        supporting_excerpt=", ".join(buyer_names) if buyer_names else None,
        reasoning_summary=evidence_note,
    )

    # Not "not_found" when there's genuinely zero buying — that's a
    # confidently-answered "No", distinct from failing to check at all.
    confident_status = FieldStatus.EXTRACTED_HIGH_CONFIDENCE

    return [
        ExtractedFieldValue(
            field_key="insider_buying_within_3mo", extraction_method="form4_aggregation",
            status=confident_status, raw_value=str(summary.any_buying),
            normalized_value=str(summary.any_buying), confidence=1.0 if not parse_failures else 0.85,
            sources=[citation],
        ),
        ExtractedFieldValue(
            field_key="cluster_insider_buying_within_3mo", extraction_method="form4_aggregation",
            status=confident_status, raw_value=str(summary.is_cluster),
            normalized_value=str(summary.is_cluster), confidence=1.0 if not parse_failures else 0.85,
            sources=[citation],
        ),
        ExtractedFieldValue(
            field_key="insider_buyer_count", extraction_method="form4_aggregation",
            status=confident_status, raw_value=str(summary.distinct_buyer_count),
            normalized_value=str(summary.distinct_buyer_count),
            numeric_value=float(summary.distinct_buyer_count),
            confidence=1.0 if not parse_failures else 0.85,
            sources=[citation],
        ),
    ]

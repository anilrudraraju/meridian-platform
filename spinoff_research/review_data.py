"""
Read-side data assembly for the review UI — joins field_values with
field_sources and field_data_dictionary display metadata into one row per
field, ready to render. Kept separate from extraction.py's
get_all_current_field_values() because the UI needs more than the raw
field_values row (display_name, category, source excerpts, review
history), and none of that belongs bundled into the extraction write path.
"""
import sqlite3
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

from spinoff_research.extraction import get_all_current_field_values
from spinoff_research.field_data_dictionary import FIELD_BY_KEY


@dataclass
class SourceView:
    document_id: Optional[int]
    supporting_excerpt: Optional[str]
    reasoning_summary: str
    sec_url: Optional[str] = None
    filing_type: Optional[str] = None


@dataclass
class FieldReviewRow:
    field_value_id: int
    field_key: str
    display_name: str
    category: str
    extraction_category: str
    status: str
    raw_value: Optional[str]
    numeric_value: Optional[float]
    confidence: Optional[float]
    as_of_date: Optional[str]
    extraction_method: Optional[str] = None  # 'xbrl' | 'filing_metadata' | 'form4_aggregation' | 'market_data' | 'ai_assisted' — from field_extraction_runs, not field_values itself
    model_used: Optional[str] = None         # populated only when extraction_method == 'ai_assisted'
    sources: List[SourceView] = dc_field(default_factory=list)


# Fields the orchestrator has a working extractor for — used to distinguish
# "not yet attempted" (no row at all) from a real not_found/uncertain result,
# and to render the other 28 fields as visibly out of scope rather than
# silently absent from the table.
DETERMINISTIC_FIELD_KEYS = {
    "form_10_availability", "parent_company_name", "spinoff_company_name",
    "parent_ticker", "spinoff_ticker", "parent_shares_outstanding",
    "spinoff_shares_outstanding", "spinoff_debt", "last_year_sales",
    "insider_buying_within_3mo", "cluster_insider_buying_within_3mo",
    "insider_buyer_count", "parent_sector", "parent_industry",
    "spinoff_sector", "spinoff_industry", "parent_share_price", "spinoff_share_price",
}


def load_review_rows(conn: sqlite3.Connection, transaction_id: int) -> List[FieldReviewRow]:
    """One FieldReviewRow per field that has ever been extracted for this
    transaction (current/non-superseded value only). Fields never
    attempted don't appear here — see DETERMINISTIC_FIELD_KEYS for the
    full set the orchestrator covers, to compute what's missing."""
    rows = get_all_current_field_values(conn, transaction_id)
    result = []
    for row in rows:
        field_def = FIELD_BY_KEY.get(row["field_key"])
        run_row = conn.execute(
            "SELECT extraction_method, model_used FROM field_extraction_runs WHERE run_id = ?",
            (row["run_id"],),
        ).fetchone()
        source_rows = conn.execute(
            """SELECT fs.*, d.sec_url, d.filing_type FROM field_sources fs
               LEFT JOIN documents d ON fs.document_id = d.document_id
               WHERE fs.field_value_id = ?""",
            (row["field_value_id"],),
        ).fetchall()
        sources = [
            SourceView(
                document_id=s["document_id"],
                supporting_excerpt=s["supporting_excerpt"],
                reasoning_summary=s["reasoning_summary"] or "",
                sec_url=s["sec_url"],
                filing_type=s["filing_type"],
            )
            for s in source_rows
        ]
        result.append(FieldReviewRow(
            field_value_id=row["field_value_id"],
            field_key=row["field_key"],
            display_name=field_def.display_name if field_def else row["field_key"],
            category=field_def.category if field_def else "unknown",
            extraction_category=field_def.extraction_category.value if field_def else "unknown",
            status=row["status"],
            raw_value=row["raw_value"],
            numeric_value=row["numeric_value"],
            confidence=row["confidence"],
            as_of_date=row["as_of_date"],
            extraction_method=run_row["extraction_method"] if run_row else None,
            model_used=run_row["model_used"] if run_row else None,
            sources=sources,
        ))
    return result


def missing_deterministic_fields(rows: List[FieldReviewRow]) -> List[str]:
    """Deterministic field_keys with no extraction attempt at all yet for
    this transaction — distinct from a field that was attempted and came
    back not_found."""
    attempted = {r.field_key for r in rows}
    return sorted(DETERMINISTIC_FIELD_KEYS - attempted)

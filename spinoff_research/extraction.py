"""
Field-agnostic persistence for extraction results.

Every deterministic extractor (XBRL, Form 4, filing metadata, yFinance —
see the per-source modules under spinoff_research/) produces a value in a
different shape internally, but they all need to land in the SAME three
tables the same way: field_extraction_runs (the attempt), field_values
(the result, with full provenance), field_sources (what backs it). This
module is that one write path, so extractors themselves stay focused on
"how do I get this value" and never reimplement persistence.

Deliberately NOT source-specific: an AI-assisted extractor (Phase 6) will
produce an ExtractedFieldValue exactly the same way a deterministic one
does today and call the same persist_field_value(). The only difference
will be extraction_method='ai_assisted' and a populated model_used.
"""
import sqlite3
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

from spinoff_research.status import FieldStatus


@dataclass
class SourceCitation:
    """One piece of evidence backing an extracted value. document_id/
    section_id are optional because some sources aren't filing documents
    at all (e.g. a yFinance API response has no document_id — the
    supporting_excerpt and reasoning_summary carry the provenance instead)."""
    document_id: Optional[int] = None
    section_id: Optional[int] = None
    supporting_excerpt: Optional[str] = None
    page_number: Optional[int] = None
    reasoning_summary: str = ""


@dataclass
class ExtractedFieldValue:
    """
    The common output shape every extractor (deterministic today,
    AI-assisted in Phase 6) produces. persist_field_value() is the only
    thing that turns this into database rows.
    """
    field_key: str
    extraction_method: str          # 'xbrl' | 'filing_metadata' | 'form4_aggregation' | 'market_data' | 'ai_assisted' | 'calculated'
    status: FieldStatus
    raw_value: Optional[str] = None
    normalized_value: Optional[str] = None
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    currency: Optional[str] = None
    scale: Optional[str] = None
    reporting_period: Optional[str] = None
    as_of_date: Optional[str] = None
    value_basis: Optional[str] = None     # 'historical' | 'pro_forma' | 'guidance' | 'calculated'
    confidence: Optional[float] = None
    model_used: Optional[str] = None      # populated only for extraction_method='ai_assisted'
    sources: List[SourceCitation] = dc_field(default_factory=list)
    selection_reason: str = ""            # carried into the run row's audit trail, not field_values itself


def persist_field_value(
    conn: sqlite3.Connection,
    transaction_id: int,
    result: ExtractedFieldValue,
) -> int:
    """
    Writes one field_extraction_runs row, one field_values row, and N
    field_sources rows (one per result.sources entry). Returns the new
    field_value_id.

    Does NOT check for or supersede a prior field_value for this
    (transaction_id, field_key) — that's a review-workflow concern
    (field_reviews / superseded_by_field_value_id), not an extraction
    concern. Calling this twice for the same field creates two independent
    field_values rows, both visible in history; only Phase 7's review UI
    decides which one is "current."
    """
    cur = conn.execute(
        """INSERT INTO field_extraction_runs
           (transaction_id, field_key, extraction_method, model_used, status)
           VALUES (?, ?, ?, ?, ?)""",
        (transaction_id, result.field_key, result.extraction_method,
         result.model_used, result.status.value),
    )
    run_id = cur.lastrowid

    cur = conn.execute(
        """INSERT INTO field_values
           (transaction_id, field_key, run_id, raw_value, normalized_value, numeric_value,
            unit, currency, scale, reporting_period, as_of_date, value_basis, confidence, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (transaction_id, result.field_key, run_id, result.raw_value, result.normalized_value,
         result.numeric_value, result.unit, result.currency, result.scale,
         result.reporting_period, result.as_of_date, result.value_basis,
         result.confidence, result.status.value),
    )
    field_value_id = cur.lastrowid

    for source in result.sources:
        conn.execute(
            """INSERT INTO field_sources
               (field_value_id, document_id, section_id, supporting_excerpt, page_number, reasoning_summary)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (field_value_id, source.document_id, source.section_id,
             source.supporting_excerpt, source.page_number, source.reasoning_summary),
        )

    conn.commit()
    return field_value_id


def get_current_field_value(conn: sqlite3.Connection, transaction_id: int, field_key: str) -> Optional[sqlite3.Row]:
    """
    Most recent field_values row for (transaction_id, field_key) that
    hasn't been superseded — the practical "current answer" for a field,
    ignoring older re-extraction attempts.
    """
    return conn.execute(
        """SELECT * FROM field_values
           WHERE transaction_id = ? AND field_key = ? AND is_original_extraction = 1
              AND superseded_by_field_value_id IS NULL
           ORDER BY created_at DESC LIMIT 1""",
        (transaction_id, field_key),
    ).fetchone()


def get_all_current_field_values(conn: sqlite3.Connection, transaction_id: int) -> List[sqlite3.Row]:
    """One row per field_key that has ever been extracted for this
    transaction, most recent non-superseded attempt only."""
    return conn.execute(
        """SELECT fv.* FROM field_values fv
           INNER JOIN (
               SELECT field_key, MAX(created_at) AS max_created
               FROM field_values
               WHERE transaction_id = ? AND superseded_by_field_value_id IS NULL
               GROUP BY field_key
           ) latest ON fv.field_key = latest.field_key AND fv.created_at = latest.max_created
           WHERE fv.transaction_id = ?""",
        (transaction_id, transaction_id),
    ).fetchall()

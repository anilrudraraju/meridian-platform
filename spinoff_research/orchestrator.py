"""
Runs every proven deterministic extractor for one transaction and persists
the results. This is the piece that closes Phase 5: previously each
extraction mechanism (XBRL, Form 4, filing metadata, yFinance) was proven
correct in isolation, but nothing called them together for a real
transaction and wrote field_values rows. run_deterministic_extraction()
is that call.

Covers the 19 fields with a proven mechanism (see field_data_dictionary.py
and the extractors/ package) — NOT all 47. AI-assisted fields (21),
calculated fields (4), and the non-Form-4 scheduled fields (tsr,
dividend_initiated_within_12mo, resources) are out of scope here; Phase 6+
work, or in calculated's case, blocked on Phase 6 fields existing first.
"""
import sqlite3
from dataclasses import dataclass
from typing import List

from spinoff_research.extraction import ExtractedFieldValue, persist_field_value
from spinoff_research.models import SpinoffTransaction
from spinoff_research.sec_service import resolve_cik
from spinoff_research.extractors.filing_metadata import extract_form_10_availability, extract_company_identity
from spinoff_research.extractors.xbrl import extract_xbrl_field
from spinoff_research.extractors.form4 import extract_insider_buying_fields
from spinoff_research.extractors.market_data import extract_sector_industry, extract_share_price


@dataclass
class ExtractionRunSummary:
    transaction_id: int
    results: List[ExtractedFieldValue]
    errors: List[str]

    @property
    def field_count(self) -> int:
        return len(self.results)

    @property
    def by_status(self) -> dict:
        counts = {}
        for r in self.results:
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        return counts


def run_deterministic_extraction(conn: sqlite3.Connection, transaction: SpinoffTransaction) -> ExtractionRunSummary:
    """
    transaction must already be persisted (transaction.transaction_id set
    — see repository.get_or_create_transaction). Requires
    transaction.parent.cik, transaction.spinoff.cik, and
    transaction.spinoff_date to be populated for most extractors to run;
    fields that need a CIK/date not yet known are skipped with an error
    logged, not silently dropped.
    """
    if transaction.transaction_id is None:
        raise ValueError("transaction must be persisted first (transaction_id is None)")

    results: List[ExtractedFieldValue] = []
    errors: List[str] = []

    parent_cik = transaction.parent.cik
    spinoff_cik = transaction.spinoff.cik
    dist_date = transaction.spinoff_date
    ann_date = transaction.announcement_date

    # ── Filing metadata (company identity + Form 10 availability) ──────────
    if transaction.parent.ticker:
        parent_resolved = resolve_cik(transaction.parent.ticker)
        results.extend(extract_company_identity(transaction.parent.ticker, parent_resolved, "parent"))
    else:
        errors.append("parent.ticker missing — skipped parent_company_name/parent_ticker")

    if transaction.spinoff.ticker:
        spinoff_resolved = resolve_cik(transaction.spinoff.ticker)
        results.extend(extract_company_identity(transaction.spinoff.ticker, spinoff_resolved, "spinoff"))
    else:
        errors.append("spinoff.ticker missing — skipped spinoff_company_name/spinoff_ticker")

    if spinoff_cik:
        results.append(extract_form_10_availability(spinoff_cik))
    else:
        errors.append("spinoff.cik missing — skipped form_10_availability")

    # ── XBRL (snapshot fields need distribution_date; last_year_sales needs announcement_date) ──
    if spinoff_cik and dist_date:
        results.append(extract_xbrl_field("spinoff_shares_outstanding", spinoff_cik, distribution_date=dist_date))
        results.append(extract_xbrl_field("spinoff_debt", spinoff_cik, distribution_date=dist_date))
    else:
        errors.append("spinoff.cik or spinoff_date missing — skipped spinoff_shares_outstanding/spinoff_debt")

    if parent_cik and dist_date:
        results.append(extract_xbrl_field("parent_shares_outstanding", parent_cik, distribution_date=dist_date))
    else:
        errors.append("parent.cik or spinoff_date missing — skipped parent_shares_outstanding")

    if spinoff_cik and ann_date:
        results.append(extract_xbrl_field("last_year_sales", spinoff_cik, announcement_date=ann_date))
    else:
        errors.append("spinoff.cik or announcement_date missing — skipped last_year_sales")

    # ── Form 4 insider buying (single aggregation pass, 3 fields) ───────────
    if spinoff_cik and dist_date:
        results.extend(extract_insider_buying_fields(spinoff_cik, distribution_date=dist_date))
    else:
        errors.append("spinoff.cik or spinoff_date missing — skipped insider buying fields")

    # ── yFinance (sector/industry + share price) ─────────────────────────────
    if transaction.parent.ticker:
        results.extend(extract_sector_industry(transaction.parent.ticker, "parent"))
    if transaction.spinoff.ticker:
        results.extend(extract_sector_industry(transaction.spinoff.ticker, "spinoff"))
    if transaction.parent.ticker and dist_date:
        results.append(extract_share_price(transaction.parent.ticker, "parent_share_price", dist_date))
    if transaction.spinoff.ticker and dist_date:
        results.append(extract_share_price(transaction.spinoff.ticker, "spinoff_share_price", dist_date))

    for result in results:
        persist_field_value(conn, transaction.transaction_id, result)

    return ExtractionRunSummary(transaction_id=transaction.transaction_id, results=results, errors=errors)

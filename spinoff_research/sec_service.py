"""
SEC EDGAR access for the spin-off research prototype.

Wraps the same public EDGAR endpoints core/edgar.py already uses
(company_tickers.json for ticker->CIK, data.sec.gov/submissions for filing
history) but does NOT reuse core/edgar.py's functions directly — those are
single-filing/single-concept helpers built for Meridian's one-document-at-a-
time RAG flow (fetch one filing's text, or one XBRL concept map). This
module needs multi-filing discovery across a date window, on-disk caching,
and CIK persistence, none of which fit core/edgar.py's shape. Reuse is at
the "same API, same conventions" level, not shared function calls.

SEC access requirements (https://www.sec.gov/os/webmaster-faq#developers):
- Descriptive User-Agent with a real contact method (SEC_EDGAR_USER_AGENT env var)
- No more than ~10 requests/second; this module paces far below that
- Cache aggressively — company_tickers.json and submissions data change
  infrequently relative to how often a research session might re-request them
"""
import json
import os
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional

import requests

from spinoff_research.field_data_dictionary import SourceType

_CACHE_DIR = Path(__file__).parent / "data" / "sec_cache"
_TICKER_MAP_CACHE = _CACHE_DIR / "company_tickers.json"
_TICKER_MAP_TTL_SECONDS = 24 * 60 * 60  # ticker/CIK map changes rarely; refresh daily at most

_REQUEST_MIN_INTERVAL_SECONDS = 0.15  # well under SEC's ~10 req/s guidance
_MAX_RETRIES = 3
_RETRY_BACKOFF_SECONDS = 1.5

_last_request_time = 0.0


class SecServiceError(Exception):
    pass


def _user_agent() -> str:
    ua = os.environ.get("SEC_EDGAR_USER_AGENT")
    if not ua:
        raise SecServiceError(
            "SEC_EDGAR_USER_AGENT is not set. SEC requires a descriptive User-Agent "
            "with a real contact method for all requests — see .env.example."
        )
    return ua


def _paced_get(url: str, timeout: int = 20) -> requests.Response:
    """GET with minimum inter-request spacing and retry-with-backoff on failure/429/5xx."""
    global _last_request_time
    headers = {"User-Agent": _user_agent()}

    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        elapsed = time.monotonic() - _last_request_time
        if elapsed < _REQUEST_MIN_INTERVAL_SECONDS:
            time.sleep(_REQUEST_MIN_INTERVAL_SECONDS - elapsed)
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            _last_request_time = time.monotonic()
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                continue
            resp.raise_for_status()
        except requests.RequestException as e:
            last_exc = e
            _last_request_time = time.monotonic()
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
                continue
    raise SecServiceError(f"GET {url} failed after {_MAX_RETRIES} attempts: {last_exc}")


# ─────────────────────────────────────────────────────────────────────────
# CIK resolution
# ─────────────────────────────────────────────────────────────────────────

def _load_ticker_map(force_refresh: bool = False) -> Dict[str, dict]:
    """
    Returns {TICKER: {"cik": "0001234567", "name": "..."}}.
    Cached on disk; company_tickers.json is ~10k entries and rarely changes,
    so re-fetching it on every CIK lookup would be wasteful and slow.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not force_refresh and _TICKER_MAP_CACHE.exists():
        age = time.time() - _TICKER_MAP_CACHE.stat().st_mtime
        if age < _TICKER_MAP_TTL_SECONDS:
            try:
                raw = json.loads(_TICKER_MAP_CACHE.read_text())
                return raw
            except (json.JSONDecodeError, OSError):
                pass  # fall through to re-fetch on a corrupt cache file

    resp = _paced_get("https://www.sec.gov/files/company_tickers.json")
    raw_entries = resp.json()
    ticker_map = {
        entry["ticker"].upper(): {
            "cik": str(entry["cik_str"]).zfill(10),
            "name": entry["title"],
        }
        for entry in raw_entries.values()
    }
    _TICKER_MAP_CACHE.write_text(json.dumps(ticker_map))
    return ticker_map


def resolve_cik(ticker: str, force_refresh: bool = False) -> Optional[dict]:
    """Returns {"cik": "0001234567", "name": "..."} or None if the ticker isn't found."""
    ticker_map = _load_ticker_map(force_refresh=force_refresh)
    return ticker_map.get(ticker.upper())


# ─────────────────────────────────────────────────────────────────────────
# Filing discovery
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class DiscoveredFiling:
    cik: str
    company_name: str
    form_type: str
    filing_date: str
    accession_number: str          # dashless, as SEC exposes it in the index URL path
    accession_number_dashed: str   # with dashes, as SEC exposes it in submissions JSON
    primary_document: str
    period_of_report: Optional[str] = None
    source_type: Optional[SourceType] = None
    sec_url: str = ""

    def __post_init__(self):
        if not self.sec_url and self.primary_document:
            cik_int = str(int(self.cik))  # EDGAR archive paths use the un-padded CIK
            self.sec_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
                f"{self.accession_number}/{self.primary_document}"
            )

    @property
    def raw_document_url(self) -> str:
        """
        The actual raw source document, as opposed to sec_url (which may
        point at an SEC XSL-rendered HTML view). Confirmed live across all
        3 pilots: the submissions API's primaryDocument for Form 4 filings
        is always reported as 'xslF345XNN/<filename>.xml' — that path
        serves a human-readable HTML rendering of the XML, not the raw
        machine-parseable XML itself, even though it ends in .xml. The
        real raw file sits at the accession root under just its filename.
        Parsers that need machine-readable content (Form 4 XML parsing)
        must fetch this URL, not sec_url.
        """
        if "/xslF345X" in self.sec_url or "/xsl" in self.primary_document.lower():
            filename = self.primary_document.rsplit("/", 1)[-1]
            cik_int = str(int(self.cik))
            return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{self.accession_number}/{filename}"
        return self.sec_url


# Form types recognized as belonging to each SourceType the field dictionary
# references. "10-12B"/"10-12G" and their amendments are Form 10 registration
# statements; SEC doesn't use a plain "FORM 10" form-type string.
#
# 20-F/6-K are included as the foreign-private-issuer equivalents of
# 10-K/8-K — a parent filer incorporated outside the US (e.g. Sanofi, the
# INBX pilot's parent) never files a 10-K or 8-K at all, so omitting these
# would silently return zero parent filings for any FPI parent rather than
# surfacing an honest gap. Confirmed against Sanofi's real filing history:
# 17 20-F + 521 6-K filings, 0 10-K/8-K.
_FORM_TYPE_TO_SOURCE_TYPE = {
    "10-12B": SourceType.FORM_10,
    "10-12B/A": SourceType.FORM_10,
    "10-12G": SourceType.FORM_10,
    "10-12G/A": SourceType.FORM_10,
    "10-K": SourceType.FORM_10K,
    "10-K/A": SourceType.FORM_10K,
    "10-Q": SourceType.FORM_10Q,
    "10-Q/A": SourceType.FORM_10Q,
    "8-K": SourceType.FORM_8K,
    "8-K/A": SourceType.FORM_8K,
    "20-F": SourceType.FORM_10K,       # FPI annual report, plays 10-K's role
    "20-F/A": SourceType.FORM_10K,
    "6-K": SourceType.FORM_8K,         # FPI current report, plays 8-K's role
    "6-K/A": SourceType.FORM_8K,
    "DEF 14A": SourceType.DEF_14A,
    "4": SourceType.FORM_4,
}

# Filing types relevant to the spin-off research prototype (brief's
# "relevant filings may include" list, mapped to actual EDGAR form-type
# strings). Filing discovery only ever looks for these — "do not download
# every filing indiscriminately."
RELEVANT_FORM_TYPES = tuple(_FORM_TYPE_TO_SOURCE_TYPE.keys())


def get_filing_history(cik: str) -> dict:
    """Raw submissions JSON for a CIK — the 'recent' filings block."""
    cik_padded = str(cik).zfill(10)
    resp = _paced_get(f"https://data.sec.gov/submissions/CIK{cik_padded}.json")
    return resp.json()


def discover_filings(
    cik: str,
    company_name: Optional[str] = None,
    form_types: Optional[List[str]] = None,
    filed_after: Optional[str] = None,
    filed_before: Optional[str] = None,
) -> List[DiscoveredFiling]:
    """
    List filings for a CIK, filtered by form type and filing-date window.

    form_types: EDGAR form-type strings to include (defaults to
        RELEVANT_FORM_TYPES — every other form type on the filer's history,
        e.g. 424B5 prospectus supplements, SC 13G ownership filings, CORRESP
        SEC correspondence, is excluded by default rather than fetched).
    filed_after / filed_before: ISO date strings ("YYYY-MM-DD"), inclusive.
        Callers should derive this window from the transaction's known
        announcement/distribution dates plus each field's preferred lookback
        (e.g. Form 4 aggregation only needs the 3 months post-distribution).
    """
    wanted_forms = set(form_types) if form_types else set(RELEVANT_FORM_TYPES)
    sub = get_filing_history(cik)
    name = company_name or sub.get("name", "")
    filings_block = sub.get("filings", {}).get("recent", {})

    forms = filings_block.get("form", [])
    dates = filings_block.get("filingDate", [])
    accessions = filings_block.get("accessionNumber", [])
    period_ends = filings_block.get("reportDate", [])
    primary_docs = filings_block.get("primaryDocument", [])

    results: List[DiscoveredFiling] = []
    for i in range(len(forms)):
        form = forms[i]
        if form not in wanted_forms:
            continue
        filing_date = dates[i] if i < len(dates) else ""
        if filed_after and filing_date < filed_after:
            continue
        if filed_before and filing_date > filed_before:
            continue
        raw_acc = accessions[i] if i < len(accessions) else ""
        if not raw_acc:
            continue
        results.append(DiscoveredFiling(
            cik=str(cik).zfill(10),
            company_name=name,
            form_type=form,
            filing_date=filing_date,
            accession_number=raw_acc.replace("-", ""),
            accession_number_dashed=raw_acc,
            primary_document=primary_docs[i] if i < len(primary_docs) else "",
            period_of_report=period_ends[i] if i < len(period_ends) else None,
            source_type=_FORM_TYPE_TO_SOURCE_TYPE.get(form),
        ))
    return results


_BACKWARD_EXPANSION_STEP_DAYS = 30
_BACKWARD_EXPANSION_MAX_MONTHS = 24


def _discover_with_backward_expansion(
    cik: str,
    company_name: str,
    form_types: List[str],
    filed_after: Optional[str],
    filed_before: Optional[str],
) -> "BucketResult":
    """
    Try the given window first. If it comes back empty and filed_after is
    set, walk filed_after backward one month at a time (up to
    _BACKWARD_EXPANSION_MAX_MONTHS total) until a filing is found or the cap
    is hit. filed_before never moves — expanding forward past a window's
    upper bound (e.g. "12 months post-distribution" for Form 4 aggregation)
    would change what the bucket means, not just how far back it looks.

    This replaces guessing a single fixed lookback per bucket (which failed
    twice during Phase 3: GE's most-recent-10-K was 9 months pre-announcement
    against a 90-day window; Sanofi's FPI form types needed their own map
    entirely) with "start from the reasoned default, keep looking until you
    find the document," which self-corrects for transactions whose filing
    cadence doesn't match the default assumption.
    """
    from datetime import datetime, timedelta

    months_expanded = 0
    current_filed_after = filed_after
    results = discover_filings(cik, company_name, form_types=form_types,
                                filed_after=current_filed_after, filed_before=filed_before)

    while not results and current_filed_after and months_expanded < _BACKWARD_EXPANSION_MAX_MONTHS:
        dt = datetime.strptime(current_filed_after, "%Y-%m-%d") - timedelta(days=_BACKWARD_EXPANSION_STEP_DAYS)
        current_filed_after = dt.strftime("%Y-%m-%d")
        months_expanded += 1
        results = discover_filings(cik, company_name, form_types=form_types,
                                    filed_after=current_filed_after, filed_before=filed_before)

    return BucketResult(
        filings=results,
        months_expanded=months_expanded,
        effective_filed_after=current_filed_after,
        exhausted=(not results and months_expanded >= _BACKWARD_EXPANSION_MAX_MONTHS),
    )


@dataclass
class BucketResult:
    filings: List[DiscoveredFiling]
    months_expanded: int = 0            # 0 = found within the original/default window
    effective_filed_after: Optional[str] = None
    exhausted: bool = False             # True = hit the 24-month cap with nothing found; a real gap, not a window-guessing miss

    def __iter__(self):
        return iter(self.filings)

    def __len__(self):
        return len(self.filings)

    def __bool__(self):
        return bool(self.filings)

    def __getitem__(self, idx):
        return self.filings[idx]


def discover_filings_for_transaction(
    parent_cik: str,
    spinoff_cik: str,
    parent_name: str,
    spinoff_name: str,
    announcement_date: Optional[str] = None,
    distribution_date: Optional[str] = None,
) -> Dict[str, List[DiscoveredFiling]]:
    """
    Discover the filing set relevant to one spin-off transaction, windowed
    around its known dates rather than pulling each filer's entire history.

    Each bucket starts from a reasoned default window (below), then — if
    that default returns nothing — _discover_with_backward_expansion() walks
    the window's lower bound (filed_after) back one month at a time, up to
    _BACKWARD_EXPANSION_MAX_MONTHS, until a filing turns up or the cap is
    hit. The upper bound (filed_before) never moves: for buckets with a
    real observation-window meaning (e.g. Form 4 aggregation should not run
    past 12 months post-distribution), only "how far back is the document
    we expect to exist" is uncertain, not "how far forward should we look."
    Form 10 has no filed_after to begin with, so expansion is a no-op there
    by construction — it was already unbounded.

    Default windows (starting point for expansion, not a hard ceiling):
      - Spinco Form 10 / amendments: no date filter.
      - Spinco 10-K/10-Q/DEF 14A: from announcement_date through 18 months
        after distribution_date.
      - Spinco 8-K and Form 4: from 90 days before announcement_date through
        12 months after distribution_date (covers the longest scheduled
        field window in the dictionary: dividend-initiated-within-12mo).
      - Parent 8-K/6-K: same announcement-anchored window as spinco event-driven.
      - Parent 10-K/20-F: up to 15 months before announcement_date through
        announcement_date — a starting guess for "most recent annual report
        before the spin," which backward expansion now corrects for
        transactions where it's off (as GE's real 10-K was, at 9 months back).
    """
    from datetime import datetime, timedelta

    def _shift(date_str: Optional[str], days: int) -> Optional[str]:
        if not date_str:
            return None
        dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)
        return dt.strftime("%Y-%m-%d")

    pre_announcement_window = _shift(announcement_date, -90)
    parent_10k_lookback = _shift(announcement_date, -450)  # ~15 months
    post_distribution_window = _shift(distribution_date, 365) if distribution_date else None
    post_distribution_periodic_window = _shift(distribution_date, 545) if distribution_date else None  # ~18 months

    spinco_form10 = _discover_with_backward_expansion(
        spinoff_cik, spinoff_name, form_types=["10-12B", "10-12B/A", "10-12G", "10-12G/A"],
        filed_after=None, filed_before=None,
    )
    spinco_periodic = _discover_with_backward_expansion(
        spinoff_cik, spinoff_name, form_types=["10-K", "10-K/A", "10-Q", "10-Q/A", "DEF 14A"],
        filed_after=announcement_date, filed_before=post_distribution_periodic_window,
    )
    spinco_event_driven = _discover_with_backward_expansion(
        spinoff_cik, spinoff_name, form_types=["8-K", "8-K/A", "4"],
        filed_after=pre_announcement_window, filed_before=post_distribution_window,
    )
    parent_event_driven = _discover_with_backward_expansion(
        parent_cik, parent_name, form_types=["8-K", "8-K/A", "6-K", "6-K/A"],
        filed_after=pre_announcement_window, filed_before=post_distribution_window,
    )
    parent_periodic = _discover_with_backward_expansion(
        parent_cik, parent_name, form_types=["10-K", "10-K/A", "20-F", "20-F/A"],
        filed_after=parent_10k_lookback, filed_before=announcement_date,
    )

    return {
        "spinco_form_10": spinco_form10,
        "spinco_periodic": spinco_periodic,
        "spinco_event_driven": spinco_event_driven,
        "parent_event_driven": parent_event_driven,
        "parent_periodic": parent_periodic,
    }

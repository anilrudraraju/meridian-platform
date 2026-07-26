"""
Retrieval helper for AI-assisted extraction over Form 10 prose: finds (and
ingests, if not already cached) the spinco's Form 10, then narrows its
document_sections down to the ones actually relevant to a search-term-based
query — e.g. "Management" section biography paragraphs — so the AI
extractor's prompt carries only a few thousand tokens of targeted prose,
not the entire filing.

Deliberately separate from ingestion.py (which handles download/cache/
section-splitting for ANY filing) — this module is the read side, specific
to "find the Form 10 for this transaction and search its sections."

IMPORTANT (found via live validation against all 3 pilots): the Form 10
primary document itself is almost always just a cover page + exhibit
index — it incorporates Item 5 (Directors and Executive Officers, where
CEO/CFO biographies live) BY REFERENCE to Exhibit 99.1, the Information
Statement, which is a separate file in the same accession. Any extractor
reading only the primary document will find zero biographical content.
get_or_ingest_form10() therefore also discovers and ingests that exhibit.

Exhibit 99.1 filename conventions are NOT standardized across filers —
confirmed live: 'd542465dex991.htm' (GE Vernova), 'd556103dex991.htm'
(Grail), and 'tm243190d12_exh99x1.htm' (Inhibrx, a THIRD distinct pattern,
found alongside an unrelated smaller 'ex99-2' press-release exhibit in the
same accession). _EXHIBIT_99_1_PATTERN matches all observed variants; when
more than one file matches, the largest by size wins (the Information
Statement runs several MB; press-release-style exhibits are single-digit
KB), rather than trusting name pattern alone to disambiguate.
"""
import re
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

from spinoff_research.ingestion import ingest_filing, IngestionError
from spinoff_research.sec_service import DiscoveredFiling, _paced_get, discover_filings, SecServiceError

_EXHIBIT_99_1_PATTERN = re.compile(r"ex.?99.?1\b|exh99x1\b", re.IGNORECASE)


class Form10NotFoundError(Exception):
    pass


@dataclass
class RelevantSection:
    section_id: int
    document_id: int
    content: str


def _find_exhibit_99_1_filename(filing: DiscoveredFiling) -> Optional[str]:
    """
    Queries the accession's index.json (EDGAR's structured directory
    listing) for a file matching an Exhibit 99.1 naming convention. Returns
    just the filename (not a full URL — callers need it to build a proper
    DiscoveredFiling so ingest_filing's exhibit_number/primary_document_name
    bookkeeping reflects the exhibit, not the primary document). Returns
    None (not an error) if no exhibit is found — some filers genuinely
    include Item 5 content directly in the primary document instead.
    """
    cik_int = str(int(filing.cik))
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{filing.accession_number}/index.json"
    try:
        resp = _paced_get(index_url)
        items = resp.json()["directory"]["item"]
    except (SecServiceError, KeyError, ValueError):
        return None

    candidates = [
        item for item in items
        if _EXHIBIT_99_1_PATTERN.search(item["name"]) and item["name"].lower().endswith((".htm", ".html"))
    ]
    if not candidates:
        return None

    best = max(candidates, key=lambda item: int(item.get("size") or 0))
    return best["name"]


def get_or_ingest_form10(
    conn: sqlite3.Connection,
    transaction_id: int,
    spinoff_cik: str,
    company_id: Optional[int] = None,
) -> List[int]:
    """
    Returns document_ids for the spinco's Form 10 (10-12B or 10-12G,
    preferring the most recently filed if multiple amendments exist) AND,
    if found, its Exhibit 99.1 Information Statement — ingesting either
    that isn't already in the documents table for this transaction. The
    primary document is always first in the returned list.

    Always returns at least one document_id (the primary Form 10) if a
    filing exists at all; the exhibit is best-effort and silently omitted
    if it can't be located or ingested, since some fields may still be
    answerable from the primary document alone.
    """
    existing_primary = conn.execute(
        "SELECT document_id, accession_number FROM documents WHERE transaction_id = ? "
        "AND filing_type LIKE '10-12%' ORDER BY filing_date DESC LIMIT 1",
        (transaction_id,),
    ).fetchone()
    if existing_primary:
        doc_ids = [existing_primary["document_id"]]
        existing_exhibit = conn.execute(
            "SELECT document_id FROM documents WHERE transaction_id = ? "
            "AND accession_number = ? AND primary_document_name != ? "
            "ORDER BY document_id LIMIT 1",
            (transaction_id, existing_primary["accession_number"],
             conn.execute("SELECT primary_document_name FROM documents WHERE document_id = ?",
                          (existing_primary["document_id"],)).fetchone()["primary_document_name"]),
        ).fetchone()
        if existing_exhibit:
            doc_ids.append(existing_exhibit["document_id"])
        return doc_ids

    try:
        filings = discover_filings(spinoff_cik, form_types=["10-12B", "10-12B/A", "10-12G", "10-12G/A"])
    except SecServiceError as e:
        raise Form10NotFoundError(f"Could not query EDGAR for Form 10 filings: {e}") from e
    if not filings:
        raise Form10NotFoundError(f"No Form 10 (10-12B/10-12G) filing found for CIK {spinoff_cik}")

    most_recent = sorted(filings, key=lambda f: f.filing_date)[-1]
    try:
        primary_result = ingest_filing(conn, transaction_id, company_id, most_recent)
    except IngestionError as e:
        raise Form10NotFoundError(f"Found Form 10 filing but could not download/ingest it: {e}") from e

    doc_ids = [primary_result.document_id]

    exhibit_filename = _find_exhibit_99_1_filename(most_recent)
    if exhibit_filename:
        # A distinct DiscoveredFiling (not most_recent reused) so
        # ingest_filing's primary_document_name/exhibit_number bookkeeping —
        # and the documents table's UNIQUE(accession_number,
        # primary_document_name, exhibit_number) constraint — correctly
        # reflect the exhibit file, not the cover-page document.
        exhibit_filing = DiscoveredFiling(
            cik=most_recent.cik, company_name=most_recent.company_name,
            form_type=most_recent.form_type, filing_date=most_recent.filing_date,
            accession_number=most_recent.accession_number,
            accession_number_dashed=most_recent.accession_number_dashed,
            primary_document=exhibit_filename, period_of_report=most_recent.period_of_report,
            source_type=most_recent.source_type,
        )
        try:
            exhibit_result = ingest_filing(conn, transaction_id, company_id, exhibit_filing)
            doc_ids.append(exhibit_result.document_id)
        except IngestionError:
            pass  # exhibit fetch is best-effort; primary document is still usable

    return doc_ids


def find_relevant_sections(
    conn: sqlite3.Connection,
    document_ids: List[int],
    search_terms: List[str],
    max_sections: int = 6,
) -> List[RelevantSection]:
    """
    Prose sections (section_title LIKE 'prose:%') from any of document_ids
    whose content contains at least one of search_terms (case-insensitive),
    ranked by how many distinct terms matched (most-relevant first), capped
    at max_sections so the extractor's prompt stays bounded regardless of
    filing length. Table sections are excluded — Management-section
    biographies are prose, not tabular data.

    Accepts multiple document_ids because a Form 10's Item 5 (management
    biographies) usually lives in a separate Exhibit 99.1 document, not the
    primary filing itself — see get_or_ingest_form10's docstring.
    """
    placeholders = ",".join("?" for _ in document_ids)
    rows = conn.execute(
        f"SELECT section_id, document_id, content FROM document_sections "
        f"WHERE document_id IN ({placeholders}) AND section_title LIKE 'prose:%' "
        f"ORDER BY document_id, chunk_index",
        tuple(document_ids),
    ).fetchall()

    scored = []
    lowered_terms = [t.lower() for t in search_terms]
    for row in rows:
        content_lower = row["content"].lower()
        match_count = sum(1 for t in lowered_terms if t in content_lower)
        if match_count > 0:
            scored.append((match_count, row))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        RelevantSection(section_id=row["section_id"], document_id=row["document_id"], content=row["content"])
        for _, row in scored[:max_sections]
    ]

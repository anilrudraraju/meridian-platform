"""
Document ingestion: download, cache, hash, and persist SEC filings
discovered by sec_service.py, then split them into document_sections using
html_table_parser.py (tables) and core/edgar.py's _strip_html (prose).

This is the piece that was missing after Phase 3: sec_service.py discovers
filings and returns DiscoveredFiling objects in memory; html_table_parser.py
and form4_parser.py can parse content once fetched — but nothing downloaded
a filing to disk, deduped it, or wrote a documents/document_sections row.
Every prior validation (Phase 3/3.5 XBRL and table-parser work) re-fetched
live from SEC in throwaway scripts. This module is what makes extraction
results reproducible and citable against a stored copy, not a live URL
that could change or 404 later.

Explicitly NOT handled here:
  - XBRL facts: pulled from the companyfacts API (xbrl_service.py), not
    from a filing document at all — no document row needed for those.
  - Form 4 XML: parsed directly (form4_parser.py) but not split into
    document_sections — it's structured data with no "prose vs table"
    distinction; the raw XML is cached and the documents row is written,
    but section-splitting doesn't apply to it.
"""
import hashlib
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests

from core.edgar import _strip_html
from spinoff_research.html_table_parser import extract_tables
from spinoff_research.sec_service import DiscoveredFiling, _paced_get, SecServiceError

_CACHE_DIR = Path(__file__).parent / "data" / "filing_cache"

# Below this size, prose paragraphs get grouped into one section rather
# than one row per paragraph — keeps document_sections from exploding into
# thousands of tiny rows for a long filing while still giving each section
# a citable char_start/char_end range.
_PROSE_CHUNK_CHARS = 4000


class IngestionError(Exception):
    pass


@dataclass
class IngestResult:
    document_id: int
    content_hash: str
    was_cached: bool          # True if this exact content was already ingested (dedup hit)
    section_count: int
    ingestion_status: str      # 'downloaded' | 'failed'
    text_extraction_status: str  # 'extracted' | 'failed' | 'not_applicable'


def _cache_path_for_hash(content_hash: str, suffix: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{content_hash}{suffix}"


def _download_raw(url: str) -> bytes:
    try:
        resp = _paced_get(url)
    except SecServiceError as e:
        raise IngestionError(f"Download failed for {url}: {e}") from e
    return resp.content


def _find_existing_document(conn: sqlite3.Connection, content_hash: str) -> Optional[sqlite3.Row]:
    """
    Dedup key is content_hash, not accession/filename — two different
    filings can legitimately reference byte-identical content (rare but
    possible with amendments that re-file an unchanged exhibit), and this
    ensures ingestion never re-downloads or re-sections something already
    on disk, regardless of which transaction or accession referenced it
    first.
    """
    return conn.execute(
        "SELECT * FROM documents WHERE content_hash = ? LIMIT 1", (content_hash,)
    ).fetchone()


def ingest_filing(
    conn: sqlite3.Connection,
    transaction_id: int,
    company_id: Optional[int],
    filing: DiscoveredFiling,
    fetch_url: Optional[str] = None,
) -> IngestResult:
    """
    Download (or reuse a cache hit for) one discovered filing, persist its
    documents row, and — for HTML content — split it into document_sections.

    fetch_url overrides which URL to download from; defaults to
    filing.raw_document_url (correct for both normal filings and the Form 4
    XSL-view-path case — see sec_service.py's raw_document_url docstring).
    Form 4 callers should pass form4_parser.parse_form4_xml's result
    separately; this function stores the raw XML but does not parse it
    (see module docstring).
    """
    url = fetch_url or filing.raw_document_url
    raw_bytes = _download_raw(url)
    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    existing = _find_existing_document(conn, content_hash)
    if existing is not None:
        section_count = conn.execute(
            "SELECT COUNT(*) AS c FROM document_sections WHERE document_id = ?",
            (existing["document_id"],),
        ).fetchone()["c"]
        return IngestResult(
            document_id=existing["document_id"],
            content_hash=content_hash,
            was_cached=True,
            section_count=section_count,
            ingestion_status=existing["ingestion_status"],
            text_extraction_status=existing["text_extraction_status"],
        )

    suffix = ".xml" if url.lower().endswith(".xml") else ".htm"
    cache_path = _cache_path_for_hash(content_hash, suffix)
    if not cache_path.exists():
        cache_path.write_bytes(raw_bytes)

    exhibit_number = None
    primary_name = filing.primary_document.rsplit("/", 1)[-1]
    if "ex" in primary_name.lower():
        exhibit_number = primary_name  # best-effort; exact exhibit numbering varies by filer

    cur = conn.execute(
        """INSERT INTO documents
           (transaction_id, company_id, cik, accession_number, filing_type, filing_date,
            period_of_report, sec_url, primary_document_name, exhibit_number, local_path,
            content_hash, ingestion_status, text_extraction_status, source_type)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'downloaded', 'pending', ?)""",
        (
            transaction_id, company_id, filing.cik, filing.accession_number_dashed,
            filing.form_type, filing.filing_date, filing.period_of_report, filing.sec_url,
            filing.primary_document, exhibit_number, str(cache_path), content_hash,
            filing.source_type.value if filing.source_type else "",
        ),
    )
    conn.commit()
    document_id = cur.lastrowid

    section_count = 0
    text_extraction_status = "not_applicable"
    if suffix == ".htm":
        try:
            html_text = raw_bytes.decode("utf-8", errors="replace")
            section_count = _extract_and_store_sections(conn, document_id, html_text)
            text_extraction_status = "extracted"
        except Exception:
            text_extraction_status = "failed"
        conn.execute(
            "UPDATE documents SET text_extraction_status = ? WHERE document_id = ?",
            (text_extraction_status, document_id),
        )
        conn.commit()

    return IngestResult(
        document_id=document_id,
        content_hash=content_hash,
        was_cached=False,
        section_count=section_count,
        ingestion_status="downloaded",
        text_extraction_status=text_extraction_status,
    )


def _extract_and_store_sections(conn: sqlite3.Connection, document_id: int, html_text: str) -> int:
    """
    Two independent passes over the same HTML, stored as two kinds of
    document_sections rows (distinguished by section_title prefix):
      - 'table:<caption>' — one row per ParsedTable, content is
        to_llm_text() (per-row, unambiguous column attribution — see
        html_table_parser.py). This is what AI-assisted extraction should
        search FIRST for any field whose value lives in a financial table.
      - 'prose:<offset>' — the stripped-HTML prose, chunked into
        _PROSE_CHUNK_CHARS windows with real char_start/char_end so a
        citation can point back to an approximate location. Table content
        is NOT re-included here (extract_tables' tables are left in place
        in the stripped text, so this is a deliberate acceptance of
        overlap rather than a second attempt to strip tables out of prose —
        precise de-duplication isn't worth the complexity it would add).
    """
    chunk_index = 0

    tables = extract_tables(html_text)
    for table in tables:
        content = table.to_llm_text()
        if not content.strip():
            continue
        title = f"table:{table.caption or 'untitled'}"
        conn.execute(
            "INSERT INTO document_sections (document_id, section_title, chunk_index, content) "
            "VALUES (?, ?, ?, ?)",
            (document_id, title, chunk_index, content),
        )
        chunk_index += 1

    prose = _strip_html(html_text)
    for start in range(0, len(prose), _PROSE_CHUNK_CHARS):
        chunk = prose[start:start + _PROSE_CHUNK_CHARS]
        if not chunk.strip():
            continue
        conn.execute(
            "INSERT INTO document_sections "
            "(document_id, section_title, chunk_index, char_start, char_end, content) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (document_id, f"prose:{start}", chunk_index, start, start + len(chunk), chunk),
        )
        chunk_index += 1

    conn.commit()
    return chunk_index

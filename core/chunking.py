import re
import streamlit as st
from typing import List, Dict, Tuple, Optional

from core.constants import (
    SECTION_PATTERNS_10K, SECTION_PATTERNS_FORM10_EXTRA, SECTION_PATTERNS_10Q,
    STATEMENT_PATTERNS, MDNA_SUBSECTION_PATTERNS,
)


def get_chunking_config(form_type: str) -> dict:
    # Normalise UI label "Form 10" → internal key "10"
    if form_type == "Form 10":
        form_type = "10"
    configs = {
        "10-K": {"business_chunk_size": 1500, "business_overlap": 300,
                 "mdna_chunk_size": 2000, "mdna_overlap": 400,
                 "skip_financial_stmts": True, "xbrl_expected": True,
                 "has_extra_sections": False, "check_predecessor": False},
        "10-Q": {"business_chunk_size": 1500, "business_overlap": 300,
                 "mdna_chunk_size": 1500, "mdna_overlap": 300,
                 "skip_financial_stmts": True, "xbrl_expected": True,
                 "has_extra_sections": False, "check_predecessor": False},
        "10":   {"business_chunk_size": 2500, "business_overlap": 500,
                 "mdna_chunk_size": 2000, "mdna_overlap": 400,
                 "skip_financial_stmts": False, "xbrl_expected": False,
                 "has_extra_sections": True, "check_predecessor": True},
    }
    return configs.get(form_type, configs["10-K"])


def _parse_filename_metadata(filename: str) -> dict:
    """Best-effort extraction of ticker, form_type, fiscal_year, quarter from filename."""
    meta = {"ticker": None, "form_type": None, "fiscal_year": None, "quarter": None}
    name = re.sub(r'\.(pdf|txt)$', '', filename, flags=re.IGNORECASE).upper()

    if re.search(r'10.?K', name):   meta["form_type"] = "10-K"
    elif re.search(r'10.?Q', name): meta["form_type"] = "10-Q"
    elif re.search(r'\b10\b', name): meta["form_type"] = "10"

    m = re.search(r'(20\d{2})', name)
    if m: meta["fiscal_year"] = m.group(1)

    # Use lookahead instead of \b — underscore is \w so Q2_ has no word boundary
    q = re.search(r'Q([123])(?=[_\-\s]|$)', name)
    if q: meta["quarter"] = f"Q{q.group(1)}"

    t = re.match(r'^([A-Z]{1,5})[_\-\s]', name)
    if t: meta["ticker"] = t.group(1)

    return meta


def _detect_fiscal_year_end(text: str) -> Optional[str]:
    """Return MM-DD fiscal year end from cover page text, or None."""
    cover = text[:3000]
    for pattern in [
        r'fiscal\s+year\s+end(?:ed|ing)\s+(\w+\s+\d{1,2},?\s*\d{4})',
        r'year\s+ended\s+(\w+\s+\d{1,2},?\s*\d{4})',
    ]:
        m = re.search(pattern, cover, re.IGNORECASE)
        if m:
            from datetime import datetime
            for fmt in ["%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"]:
                try:
                    d = datetime.strptime(m.group(1).strip(), fmt)
                    return f"{d.month:02d}-{d.day:02d}"
                except ValueError:
                    continue
    return None


def _detect_quarter_from_text(text: str, fiscal_year_end_mmdd: Optional[str]) -> Optional[str]:
    """Map period-end date on cover page to Q1/Q2/Q3 using fiscal year end as anchor."""
    cover = text[:3000]
    period_date = None
    for pattern in [
        r'(?:period|quarter)\s+ended\s+(\w+\s+\d{1,2},?\s*\d{4})',
        r'three\s+months\s+ended\s+(\w+\s+\d{1,2},?\s*\d{4})',
        r'nine\s+months\s+ended\s+(\w+\s+\d{1,2},?\s*\d{4})',
    ]:
        m = re.search(pattern, cover, re.IGNORECASE)
        if m:
            from datetime import datetime
            for fmt in ["%B %d, %Y", "%B %d %Y", "%b %d, %Y"]:
                try:
                    period_date = datetime.strptime(m.group(1).strip(), fmt)
                    break
                except ValueError:
                    continue
        if period_date:
            break
    if not period_date:
        return None
    fy_month = int(fiscal_year_end_mmdd.split("-")[0]) if fiscal_year_end_mmdd else 12
    months_from_fy_end = (period_date.month - fy_month) % 12
    return {3: "Q1", 6: "Q2", 9: "Q3"}.get(months_from_fy_end)


def _split_into_sections(text: str, form_type: str) -> Dict[str, str]:
    """Split full filing text into {section_name: section_text} using SEC Item headers."""
    # Normalise UI label "Form 10" → internal key "10"
    if form_type == "Form 10":
        form_type = "10"
    if form_type == "10-Q":
        patterns = SECTION_PATTERNS_10Q
    else:
        patterns = dict(SECTION_PATTERNS_10K)
        if form_type == "10":
            patterns.update(SECTION_PATTERNS_FORM10_EXTRA)

    matches = []
    for section_name, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            matches.append((m.start(), section_name))

    if not matches:
        return {"general": text}

    matches.sort(key=lambda x: x[0])
    sections = {}
    for i, (start, name) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections[name] = section_text
    return sections


def _chunk_by_paragraphs(text: str, chunk_size: int, overlap: int,
                          min_length: int = 100) -> List[str]:
    """Split on double newlines first; fall back to character sliding window."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if len(current) >= min_length:
                chunks.append(current)
            if len(para) > chunk_size:
                step = max(1, chunk_size - overlap)
                for j in range(0, len(para), step):
                    piece = para[j:j + chunk_size]
                    if len(piece) >= min_length:
                        chunks.append(piece)
                current = ""
            else:
                current = para
    if len(current) >= min_length:
        chunks.append(current)
    return chunks


def _chunk_business(text: str, config: dict, base_meta: dict) -> List[Dict]:
    pieces = _chunk_by_paragraphs(text, config["business_chunk_size"],
                                  config["business_overlap"], min_length=100)
    return [{"page_content": p,
             "metadata": {**base_meta, "section": "business", "chunk_id": i}}
            for i, p in enumerate(pieces)]


def _chunk_risk_factors(text: str, config: dict, base_meta: dict) -> List[Dict]:
    # 10-Q minimal risk section — store as single chunk
    if len(text.strip()) < 300:
        return [{"page_content": text.strip(),
                 "metadata": {**base_meta, "section": "risk_factors",
                               "chunk_id": 0, "risk_update": "no_material_changes",
                               "risk_header": ""}}]
    # Detect per-risk headers: ALL CAPS or Title Case, 15-100 chars, own line
    header_re = re.compile(
        r'(?m)^[ \t]*([A-Z][A-Z\s,\-\']{14,98}[A-Z]'
        r'|(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,}))[ \t]*$'
    )
    matches = list(header_re.finditer(text))
    if len(matches) < 2:
        pieces = _chunk_by_paragraphs(text, 1500, 300, min_length=150)
        return [{"page_content": p,
                 "metadata": {**base_meta, "section": "risk_factors",
                               "chunk_id": i, "risk_header": "", "risk_update": ""}}
                for i, p in enumerate(pieces)]
    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        piece = text[m.start():end].strip()
        if len(piece) >= 150:
            chunks.append({"page_content": piece,
                           "metadata": {**base_meta, "section": "risk_factors",
                                         "chunk_id": i, "risk_update": "",
                                         "risk_header": m.group(1).strip()[:100]}})
    return chunks


def _chunk_mdna(text: str, config: dict, base_meta: dict) -> List[Dict]:
    sub_matches = []
    for name, pattern in MDNA_SUBSECTION_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            sub_matches.append((m.start(), name))
    sub_matches.sort(key=lambda x: x[0])

    if not sub_matches:
        pieces = _chunk_by_paragraphs(text, config["mdna_chunk_size"],
                                      config["mdna_overlap"], min_length=100)
        return [{"page_content": p,
                 "metadata": {**base_meta, "section": "mdna",
                               "subsection": "", "chunk_id": i}}
                for i, p in enumerate(pieces)]

    chunks, chunk_id = [], 0
    for i, (start, sub_name) in enumerate(sub_matches):
        end = sub_matches[i + 1][0] if i + 1 < len(sub_matches) else len(text)
        sub_text = text[start:end].strip()
        if len(sub_text) < 100:
            continue
        for piece in _chunk_by_paragraphs(sub_text, config["mdna_chunk_size"],
                                           config["mdna_overlap"], min_length=100):
            chunks.append({"page_content": piece,
                           "metadata": {**base_meta, "section": "mdna",
                                         "subsection": sub_name, "chunk_id": chunk_id}})
            chunk_id += 1
    return chunks


def _detect_statement_boundaries(text: str) -> List[Tuple[int, str]]:
    """Find financial statement headers near line starts (within 80 chars of newline)."""
    boundaries, seen = [], set()
    for stmt_type, pattern in STATEMENT_PATTERNS.items():
        for m in re.finditer(pattern, text, re.IGNORECASE):
            preceding = text[max(0, m.start() - 80):m.start()]
            if ('\n' in preceding or m.start() < 80) and stmt_type not in seen:
                boundaries.append((m.start(), stmt_type))
                seen.add(stmt_type)
                break
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def _scan_audit_status(chunk_text: str, period_end: str) -> Tuple[bool, str]:
    """Scan first 300 chars of a financial statement chunk for audit status (Form 10 only)."""
    window = chunk_text[:300].lower()
    if "unaudited" in window:
        return False, "unaudited interim period"
    if "predecessor" in window:
        return True, "predecessor period audited"
    # Secondary: mid-year period end without standard quarter-end month
    if period_end:
        try:
            from datetime import date
            d = date.fromisoformat(period_end)
            if d.month not in (3, 6, 9, 12):
                return True, "audit status uncertain"
        except ValueError:
            pass
    if ("see accompanying notes" not in chunk_text.lower()
            and "see notes" not in chunk_text.lower()
            and len(chunk_text) < 1000):
        return True, "audit status uncertain"
    return True, "current period audited"


def _chunk_financial_stmts(text: str, config: dict, base_meta: dict,
                            xbrl_available: bool) -> List[Dict]:
    if config["skip_financial_stmts"] and xbrl_available:
        return []

    boundaries = _detect_statement_boundaries(text)
    if not boundaries:
        st.warning("⚠️ Could not detect individual financial statement boundaries in Item 8 — storing as single block.")
        meta = {**base_meta, "section": "financial_stmts", "subsection": "general",
                "statement_type": "general", "atomic_chunk": True,
                "data_source": "pdf_extracted"}
        return [{"page_content": text[:6000], "metadata": meta}]

    chunks, chunk_id = [], 0
    form_type = base_meta.get("form_type", "10-K")

    for i, (start, stmt_type) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        stmt_text = text[start:end].strip()

        if len(stmt_text) < 500 or len(re.findall(r'\d+', stmt_text)) < 5:
            continue  # TOC false positive

        if form_type in ("10", "Form 10"):
            audited, audit_note = _scan_audit_status(stmt_text, base_meta.get("period_end", ""))
            window = stmt_text[:500].lower()
            period_type = ("predecessor" if "predecessor" in window
                           else "successor" if any(w in window for w in
                                                   ("successor", "fresh-start", "reorganization"))
                           else "")
        else:
            audited = form_type == "10-K"
            audit_note = "current period audited" if audited else "unaudited interim period"
            period_type = ""

        base_stmt = {**base_meta, "section": "financial_stmts",
                     "statement_type": stmt_type, "audited": audited,
                     "audit_note": audit_note, "period_type": period_type,
                     "data_source": "pdf_extracted" if not xbrl_available else "xbrl"}

        if len(stmt_text) <= 5800:
            chunks.append({"page_content": stmt_text,
                           "metadata": {**base_stmt, "chunk_id": chunk_id,
                                         "atomic_chunk": True}})
            chunk_id += 1
        else:
            parts = [p.strip() for p in re.split(r'\n\s*\n', stmt_text) if p.strip()]
            part_chunks, current = [], ""
            for part in parts:
                if len(current) + len(part) + 2 <= 5800:
                    current = (current + "\n\n" + part).strip() if current else part
                else:
                    if current:
                        part_chunks.append(current)
                    current = part
            if current:
                part_chunks.append(current)

            total = len(part_chunks)
            for j, part_text in enumerate(part_chunks):
                content = part_text if j == 0 else \
                    f"[Continued: {stmt_type.replace('_', ' ').title()}]\n\n{part_text}"
                chunks.append({"page_content": content,
                               "metadata": {**base_stmt, "chunk_id": chunk_id,
                                             "atomic_chunk": False,
                                             "part_number": j + 1,
                                             "total_parts": total}})
                chunk_id += 1
    return chunks


def _chunk_footnotes(text: str, config: dict, base_meta: dict) -> List[Dict]:
    note_re = re.compile(r'(?m)^[ \t]*(?:Note\s+(\d+)|(\d+)\.)[ \t]+\S', re.IGNORECASE)
    matches = list(note_re.finditer(text))
    if len(matches) < 2:
        pieces = _chunk_by_paragraphs(text, 1000, 200, min_length=150)
        return [{"page_content": p,
                 "metadata": {**base_meta, "section": "footnotes",
                               "chunk_id": i, "footnote_number": 0}}
                for i, p in enumerate(pieces)]
    chunks = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        piece = text[m.start():end].strip()
        note_num = int(m.group(1) or m.group(2))
        if len(piece) >= 150:
            chunks.append({"page_content": piece,
                           "metadata": {**base_meta, "section": "footnotes",
                                         "chunk_id": i, "footnote_number": note_num}})
    return chunks


def _chunk_default(text: str, chunk_size: int, overlap: int,
                   section_name: str, base_meta: dict) -> List[Dict]:
    pieces = _chunk_by_paragraphs(text, chunk_size, overlap, min_length=100)
    return [{"page_content": p,
             "metadata": {**base_meta, "section": section_name, "chunk_id": i}}
            for i, p in enumerate(pieces)]


def _chunk_all_sections(sections: Dict[str, str], config: dict,
                         base_meta: dict, xbrl_available: bool) -> List[Dict]:
    """Dispatch each section to its chunking strategy; return all chunks."""
    all_chunks = []
    for section_name, section_text in sections.items():
        if not section_text.strip():
            continue
        meta = {**base_meta, "subsection": "", "risk_header": "",
                "footnote_number": 0, "statement_type": "", "atomic_chunk": True,
                "part_number": 1, "total_parts": 1, "risk_update": "",
                "period_type": "", "audit_note": "", "audited": True}
        if section_name == "business":
            chunks = _chunk_business(section_text, config, meta)
        elif section_name == "risk_factors":
            chunks = _chunk_risk_factors(section_text, config, meta)
        elif section_name == "mdna":
            chunks = _chunk_mdna(section_text, config, meta)
        elif section_name == "financial_stmts":
            chunks = _chunk_financial_stmts(section_text, config, meta, xbrl_available)
        elif section_name == "footnotes":
            chunks = _chunk_footnotes(section_text, config, meta)
        else:
            chunks = _chunk_default(section_text, 1000, 200, section_name, meta)
        all_chunks.extend(chunks)
    return all_chunks

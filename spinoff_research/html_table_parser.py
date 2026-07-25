"""
Table-aware HTML parsing for SEC filings — specifically hardened against
Form 10 Information Statements (Exhibit 99.1), the single most important
document type in this project.

Why this exists: Meridian's core/edgar.py._strip_html() flattens HTML to
plain text uniformly, which is fine for prose (risk factors, business
description) but silently corrupts financial tables. Confirmed live against
GRAL's actual Form 10 (d556103dex991.htm) — a 4-column Successor/Predecessor
income statement collapses to a bare sequence of numbers with no per-row
column labels:
    "Screening revenue $ 74,347 $ 39,123 $ 7,074 $ 1,953"
By row 20 of that table there is no way to recover which number belongs to
which of the four reporting periods without recounting column position —
exactly the failure mode that would corrupt last_year_sales / spinoff_debt
extraction for any Form 10 with a Successor/Predecessor split (common
whenever there's a goodwill impairment or ownership-basis change, which
GRAL has).

Also confirmed live: these tables are built from deeply nested <TD> soup
with two adversarial patterns that a naive parser must handle correctly:
  - colspan/rowspan on real header cells (357 colspan occurrences in GRAL's
    Form 10 alone) — column headers span 2 cells each, so grid position
    alone doesn't map 1:1 to a data column without expansion.
  - Large numbers of EMPTY spacer <TD>&nbsp;</TD> cells used purely for
    visual whitespace between columns — naive colspan expansion without
    collapsing these produces dozens of phantom empty columns.

Approach: expand every table into a proper grid (respecting colspan/rowspan),
collapse whitespace-only spacer columns, detect the header row(s), and
render each data row with its column headers re-attached — e.g.
"Screening revenue | Year Ended December 31, 2023 (Successor): $74,347 |
Year Ended January 1, 2023 (Predecessor): $39,123 | ..." — so column
identity survives into whatever text an LLM (or a human reviewer) reads,
instead of being positional and silently losable.
"""
import re
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

from bs4 import BeautifulSoup, Tag


_UNLABELED_HEADER = re.compile(r"^Column \d+$")


@dataclass
class ParsedTable:
    caption: Optional[str]              # nearest preceding bold/heading text, if found
    headers: List[str]                  # one label per data column, after spacer collapse
    rows: List[List[str]]               # each row's cell values, aligned to `headers`
    raw_row_labels: List[str]           # first-column text for each row (line-item label)

    @property
    def has_unlabeled_columns(self) -> bool:
        """
        True when one or more columns survived merging/collapsing without a
        recoverable header label ('Column N' placeholder). Seen in practice
        on Pro Forma adjustment tables with per-value footnote-reference
        columns (e.g. '(a)', '(b)') and some percent-change columns —
        genuinely ambiguous content, not a parser bug to keep chasing.
        Extraction consuming this table should treat it as lower-confidence
        rather than trust a fully-labeled parse.
        """
        return any(_UNLABELED_HEADER.match(h) for h in self.headers)

    def to_llm_text(self, include_unlabeled_columns: bool = False) -> str:
        """
        Render as one line per (row_label, column_header, value) triple —
        deliberately repetitive rather than a compact grid, because a
        repetitive-but-unambiguous format is safer for LLM extraction than
        a compact table an LLM might misalign. This is the format actually
        handed to extraction prompts, not the ParsedTable object itself.

        By default, columns that never resolved to a real header ('Column
        N') are OMITTED rather than shown as confusing fragments like
        'Column 5: %' — a value with no recoverable column meaning is not
        useful to an extractor and is better left out than guessed at.
        Pass include_unlabeled_columns=True for debugging/review purposes.
        """
        lines = []
        if self.caption:
            lines.append(f"TABLE: {self.caption}")
        for label, row in zip(self.raw_row_labels, self.rows):
            if not label.strip():
                continue
            parts = [
                f"{h}: {v}" for h, v in zip(self.headers, row)
                if v.strip() and (include_unlabeled_columns or not _UNLABELED_HEADER.match(h))
            ]
            if parts:
                lines.append(f"{label} — " + "; ".join(parts))
        return "\n".join(lines)


_INVISIBLE_CHARS = re.compile(r"[​‌‍﻿­]")  # zero-width space/joiner, BOM, soft hyphen


def _cell_text(cell: Tag) -> str:
    """
    Confirmed live (Inhibrx's Form 10) that some filers' templates use
    zero-width spaces (U+200B) as spacer content instead of &nbsp; —
    invisible when rendered, but NOT stripped by str.strip() (it's not
    classified as whitespace), so a "blank" spacer cell was surviving as a
    non-empty '​' string. That defeated both header-merge dedup and
    _collapse_spacer_columns, producing enormous run-on header strings
    (five real headers concatenated into one via the dedup join). Must
    strip these BEFORE whitespace collapsing, or they leave stray extra
    spaces that then get treated as content.
    """
    text = cell.get_text(separator=" ", strip=True)
    text = text.replace("\xa0", " ")
    text = _INVISIBLE_CHARS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _expand_grid(table: Tag) -> List[List[str]]:
    """Expand a <table> into a rectangular grid of cell text. Thin wrapper
    around _expand_grid_with_spans() for callers that don't need span info."""
    text_grid, _ = _expand_grid_with_spans(table)
    return text_grid


def _expand_grid_with_spans(table: Tag) -> tuple:
    """
    Expand a <table> into a rectangular grid of cell text, respecting
    colspan/rowspan, AND a parallel grid recording each cell's original
    colspan width at every position it covers. Cells spanning multiple
    columns/rows are repeated at every grid position they cover — this is
    what makes header-row lookup by column index correct later, instead of
    drifting once a rowspan cell is encountered.

    The colspan grid exists specifically to distinguish a genuine header
    row from a first DATA row whose label text happens to be non-numeric
    (confirmed live on GRAL's 10-K: a line-item row like "Gross loss (1)"
    is textually indistinguishable from a header label like "December 31,
    2024" using text content alone — but a real header row's cells span
    MULTIPLE columns as period/group labels, while a data row's label cell
    spans exactly 1 column even when its text is non-numeric).
    """
    trs = table.find_all("tr")
    grid: List[List[str]] = []
    span_grid: List[List[int]] = []
    pending: dict = {}

    for r_idx, tr in enumerate(trs):
        row_cells: List[str] = []
        row_spans: List[int] = []
        col_idx = 0

        def _next_free_col(c: int) -> int:
            while pending.get((r_idx, c), 0) > 0:
                c += 1
            return c

        for cell in tr.find_all(["td", "th"], recursive=False):
            col_idx = _next_free_col(col_idx)
            text = _cell_text(cell)
            colspan = int(cell.get("colspan", 1) or 1)
            rowspan = int(cell.get("rowspan", 1) or 1)
            for span_c in range(colspan):
                c = col_idx + span_c
                while len(row_cells) <= c:
                    row_cells.append("")
                    row_spans.append(1)
                row_cells[c] = text
                row_spans[c] = colspan
                if rowspan > 1:
                    for future_r in range(r_idx + 1, r_idx + rowspan):
                        pending[(future_r, c)] = pending.get((future_r, c), 0) + 1
            col_idx += colspan
        grid.append(row_cells)
        span_grid.append(row_spans)

    max_cols = max((len(r) for r in grid), default=0)
    grid = [r + [""] * (max_cols - len(r)) for r in grid]
    span_grid = [r + [1] * (max_cols - len(r)) for r in span_grid]
    return grid, span_grid


_CURRENCY_ONLY = re.compile(r"^[\$€£]$")
# Any short run of pure closing-punctuation: accounting-style ')', placeholder-style ']'
# (e.g. '[●]'), a split '%' sign, or combinations like '%)' (a negative percentage's
# close-paren fused with its percent sign in the same stray cell).
_CLOSE_SUFFIX_ONLY = re.compile(r"^[\)\]%]{1,3}$")


def _merge_currency_symbol_columns(grid: List[List[str]]) -> List[List[str]]:
    """
    Fold '$'-prefix and ')'/']'/'%'-suffix symbol-only cells into their
    adjacent numeric neighbor, PER ROW rather than per column.

    Originally detected this as a whole-column pattern ("this column is
    blank-or-'$' in EVERY row, merge the whole column into the next").
    That broke on standard accounting formatting confirmed live on GRAL's
    10-K: a '$' sign appears only on the first/subtotal row of a numeric
    block (e.g. "Gross loss" row 0 has '$' in a column, but "Amortization
    of intangible assets" two rows below has a REAL NUMBER '133,889' in
    that exact same column) — so the column is not a dedicated symbol
    column at all, it's a normal data column that happens to carry a
    currency prefix on some rows and not others. A column-level plan can
    never correctly handle this; only a per-row, per-cell decision can.

    Grid stays perfectly rectangular throughout — merged-away symbol cells
    become "" rather than being removed, so no row ever changes length.
    A column that ends up "" in every row after this (a true dedicated
    symbol/spacer column) gets dropped later by _collapse_spacer_columns,
    same as any other all-blank column; a column that's genuinely mixed
    (sometimes a merged '$123', sometimes a bare '456') is correctly kept.
    """
    if not grid or not grid[0]:
        return grid
    num_cols = len(grid[0])
    new_grid = [row[:] for row in grid]  # deep-enough copy: rows are lists of str

    for row in new_grid:
        # prefix pass: '$' merges INTO the next non-empty cell
        for c in range(num_cols - 1):
            val = row[c].strip()
            if val and _CURRENCY_ONLY.match(val):
                nxt = c + 1
                if row[nxt].strip():
                    row[nxt] = f"{val}{row[nxt]}"
                    row[c] = ""
        # suffix pass: ')' / ']' / '%' merges INTO the previous non-empty cell
        for c in range(1, num_cols):
            val = row[c].strip()
            if val and _CLOSE_SUFFIX_ONLY.match(val):
                prev = c - 1
                if row[prev].strip():
                    row[prev] = f"{row[prev]}{val}"
                    row[c] = ""

    return new_grid


def _nonspacer_column_indices(grid: List[List[str]]) -> List[int]:
    """Column indices that have content in at least one row — the complement
    of &nbsp;-only spacer <TD> columns. Exposed separately from
    _collapse_spacer_columns so a caller (parse_table) can apply the exact
    same column selection to a second, parallel grid (span_grid) that must
    stay column-aligned with the text grid after collapsing."""
    if not grid:
        return []
    num_cols = len(grid[0])
    return [c for c in range(num_cols) if any(row[c].strip() for row in grid)]


def _collapse_spacer_columns(grid: List[List[str]]) -> List[List[str]]:
    """
    Drop any column that is empty/whitespace across every row — the
    &nbsp;-only spacer <TD> cells SEC filing templates use for visual gaps
    between real columns. Without this, colspan-expanded grids end up with
    2-3x more columns than there are actual data series, and header/data
    row alignment breaks.
    """
    if not grid:
        return grid
    keep = _nonspacer_column_indices(grid)
    return [[row[c] for c in keep] for row in grid]


def _looks_like_header_row(row: List[str], span_row: Optional[List[int]] = None) -> bool:
    """
    A header row is mostly non-numeric text (dates, period labels) with at
    least one non-empty cell — data rows are mostly numbers/currency. This
    alone is NOT sufficient: a data row's line-item label ("Gross loss (1)")
    is also non-numeric text, and was confirmed live (GRAL's 10-K) to be
    mis-classified as a header row using text content alone, corrupting the
    header/data split for the whole table.

    A "requires colspan > 1 somewhere" signal was tried next, but also
    confirmed wrong: a legitimate simple 2-column header row with NO
    colspan at all (e.g. ['', 'Historical', 'Pro Forma'], one <td> per
    column) got rejected as non-header, breaking ordinary tables that
    don't happen to use column-grouping.

    The reliable signal: a genuine header row's own LABEL COLUMN (index 0)
    is blank — headers don't have a line-item name. A real data row's
    label column is always populated ("Gross loss (1)", "Cash", "Total
    debt(a)"). This works regardless of whether the header uses colspan,
    and is the PRIMARY check — the numeric-content ratio is only a
    fallback for the (rarer) case where column 0 isn't blank. Bare-year
    header rows like ['', '2024', '2024', '2023', '2023'] confirmed live
    to otherwise fail the numeric-content check (years look numeric), so
    that check alone is not sufficient and must not gate the more reliable
    blank-label-column signal.
    span_row is accepted for API compatibility but no longer required.
    """
    non_empty = [c for c in row if c.strip()]
    if not non_empty:
        return False
    if not row[0].strip():
        return True  # blank label column — reliable header signal on its own
    numeric_like = sum(1 for c in non_empty if re.fullmatch(r"[\$\(\)\-\d,.\s%]+", c))
    return numeric_like < len(non_empty) / 2


def _is_annotation_only_row(
    row: List[str],
    span_row: Optional[List[int]] = None,
    next_row: Optional[List[str]] = None,
) -> bool:
    """
    A row like ['', '(unaudited)', '(unaudited)', ..., '', ''] or
    ['Consolidated Statements of Operations Data:', '', '', ...] — a
    single distinct non-empty phrase repeated/isolated across the row,
    not real per-column header content. These sit between the real header
    rows and the data rows in SEC tables and must be skipped rather than
    merged into the header or mistaken for the first data row.

    Text-only "one distinct phrase" detection is NOT sufficient: confirmed
    live (GRAL's 10-K) that a genuine top-level header row — a single
    phrase like "Year Ended" spanning one wide colspan group, with the
    ACTUAL distinct period labels ("December 31, 2024" / "December 31,
    2023") appearing only in the row below it — is structurally identical
    to a true throwaway annotation like "(unaudited)" by every signal
    internal to the row itself (one distinct phrase, one contiguous span).
    Attempted a "how many separate spans" heuristic first; confirmed live
    that it's wrong too, since "Year Ended" really is one contiguous span
    here, same as "(unaudited)" — the row's own content can't disambiguate
    these two cases at all.

    The real distinguishing signal is what comes AFTER: a genuine grouping
    header like "Year Ended" is followed by a row with multiple DISTINCT
    non-empty values (the actual period dates it's grouping) — a
    throwaway annotation like "(unaudited)" is typically followed by
    either more headers or the real data rows, not a new row of distinct
    group labels. When next_row is available, only treat a single-phrase
    row as an annotation if the row below it does NOT introduce multiple
    distinct labels of its own.
    """
    non_empty = [c.strip() for c in row if c.strip()]
    if not non_empty:
        return False
    distinct = set(non_empty)
    if len(non_empty) == 1:
        return True  # exactly one non-empty cell total (a section-title row) — always annotation-only

    if len(distinct) != 1:
        return False  # multiple different phrases — not a single repeated annotation at all

    if next_row is not None:
        next_non_empty = [c.strip() for c in next_row if c.strip()]
        next_distinct = set(next_non_empty)
        if len(next_distinct) > 1:
            return False  # next row introduces multiple distinct labels — this row is a grouping header, not an annotation

    return True


def parse_table(table: Tag) -> Optional[ParsedTable]:
    """
    Parse one <table> element into a ParsedTable. Returns None for tables
    that are structurally too sparse to be meaningful (e.g. single-cell
    layout tables some filings use for spacing — not uncommon in SEC HTML).
    """
    grid, span_grid = _expand_grid_with_spans(table)
    # Apply the SAME column selection to span_grid as _collapse_spacer_columns
    # applies to grid — without this, span_grid silently drifts out of
    # column-alignment with grid after spacer columns are dropped, and
    # _looks_like_header_row ends up comparing a cell's text against the
    # WRONG column's original colspan width.
    keep_col_idx = _nonspacer_column_indices(grid)
    grid = [[row[c] for c in keep_col_idx] for row in grid]
    span_grid = [[row[c] for c in keep_col_idx] for row in span_grid]
    keep_row_idx = [i for i, row in enumerate(grid) if any(c.strip() for c in row)]
    grid = [grid[i] for i in keep_row_idx]
    span_grid = [span_grid[i] for i in keep_row_idx if i < len(span_grid)]

    if len(grid) < 2 or len(grid[0]) < 2:
        return None  # not a real multi-row/multi-column data table

    # Header detection: SEC financial tables often use 1-2 real header rows
    # (a super-header like "(Successor)/(Predecessor)" plus period dates
    # below it), frequently followed by an annotation-only row like
    # "(unaudited)" or a section-title row before real data starts. Walk
    # forward collecting header rows, skipping annotation rows, and stop at
    # the first row that looks like real data.
    header_rows = []
    data_start = 0
    for i, row in enumerate(grid):
        if i >= 5:  # header block only ever expected near the very top
            break
        span_row = span_grid[i] if i < len(span_grid) else None
        next_row = grid[i + 1] if i + 1 < len(grid) else None
        if _is_annotation_only_row(row, span_row, next_row):
            data_start = i + 1
            continue
        if _looks_like_header_row(row, span_row):
            header_rows.append(row)
            data_start = i + 1
        else:
            break

    if not header_rows:
        return None  # can't identify column structure — safer to skip than mis-attribute

    # Currency-symbol merge only makes sense once header rows are excluded —
    # header cells like "(Successor)" would otherwise be mistaken for stray
    # symbol cells. The merge is now per-row/per-cell and always preserves
    # column count (see _merge_currency_symbol_columns), so header_rows
    # never needs realignment afterward — grid stays rectangular throughout.
    data_rows_only = grid[data_start:]
    merged_data_rows = _merge_currency_symbol_columns(data_rows_only)

    num_cols = len(merged_data_rows[0]) if merged_data_rows else len(grid[0])
    combined_headers = []
    for c in range(num_cols):
        parts = [row[c] for row in header_rows if c < len(row) and row[c].strip()]
        combined_headers.append(" ".join(dict.fromkeys(parts)))  # dedupe repeated spans, preserve order

    # The row-label column itself is frequently colspan>1 after expansion
    # (confirmed live: GRAL's 10-K line-item column — e.g. "Gross loss (1)"
    # — spans 3 raw grid columns, not 1). Treating only index 0 as the
    # label and columns 1-2 as real DATA columns produced a spurious
    # leftover "(in thousands)" header duplicated three times and bled the
    # row label itself into the first data column's values. Detect the
    # label column's true width from the first data row's own span info
    # (pre-merge span_grid, since the currency merge doesn't touch the
    # label column) and skip all of it, not just index 0.
    label_col_width = 1
    first_data_row_idx = data_start
    if first_data_row_idx < len(span_grid):
        label_col_width = max(1, span_grid[first_data_row_idx][0] if span_grid[first_data_row_idx] else 1)

    row_label_col = 0
    candidate_data_cols = list(range(label_col_width, num_cols))

    # A header group spanning colspan>1 (e.g. "Year Ended December 31, 2024"
    # repeated across 3 raw columns) usually has real data in only ONE of
    # those raw columns per data row — the others are $-prefix/spacer
    # remnants the currency merge and spacer-collapse already tried to
    # clear, but couldn't fully resolve when the header text itself (not
    # just symbols) got repeated across the span. Keep a data column only
    # if it has a value in at least one actual data row — collapsing
    # against merged_data_rows specifically, not the whole pre-merge grid,
    # since a column can be legitimately blank in the header rows but still
    # be the one real data column for its period.
    data_cols = [
        c for c in candidate_data_cols
        if any(c < len(row) and row[c].strip() for row in merged_data_rows)
    ]
    if not data_cols:
        data_cols = candidate_data_cols  # degenerate case: no data rows at all — keep original columns rather than produce an empty table

    headers = [combined_headers[c] or f"Column {c}" for c in data_cols]

    rows: List[List[str]] = []
    raw_row_labels: List[str] = []
    for row in merged_data_rows:
        raw_row_labels.append(row[row_label_col] if row else "")
        rows.append([row[c] if c < len(row) else "" for c in data_cols])

    caption = None
    prev = table.find_previous(["b", "strong", "h1", "h2", "h3", "h4"])
    if prev:
        cap_text = _cell_text(prev)
        # Require actual heading-like text: a real word, not a stray bolded
        # symbol from an adjacent table's data cell (confirmed live: a bare
        # '%' from a neighboring table's percent-change column was picked
        # up as a caption on an unrelated table).
        if cap_text and 3 <= len(cap_text) < 150 and re.search(r"[A-Za-z]{2,}", cap_text):
            caption = cap_text

    return ParsedTable(caption=caption, headers=headers, rows=rows, raw_row_labels=raw_row_labels)


def extract_tables(html: str, min_rows: int = 2) -> List[ParsedTable]:
    """Parse every structurally-real table out of a filing's raw HTML."""
    soup = BeautifulSoup(html, "lxml")
    results = []
    for table in soup.find_all("table"):
        parsed = parse_table(table)
        if parsed and len(parsed.rows) >= min_rows:
            results.append(parsed)
    return results

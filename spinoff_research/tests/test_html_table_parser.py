import unittest

from spinoff_research.html_table_parser import extract_tables, parse_table
from bs4 import BeautifulSoup


def _table(html_fragment: str):
    """Wrap a <table>...</table> fragment and return the parsed BeautifulSoup <table> tag."""
    soup = BeautifulSoup(f"<html><body>{html_fragment}</body></html>", "lxml")
    return soup.find("table")


class TestSimpleTwoColumnTable(unittest.TestCase):
    """Regression fixture: GEV's real CAPITALIZATION table structure."""

    HTML = """
    <b>CAPITALIZATION</b>
    <table>
      <tr><td></td><td colspan="2" align="center">As of December 31, 2023</td></tr>
      <tr><td>($ in millions)</td><td>Historical</td><td>Pro Forma</td></tr>
      <tr><td>Total debt(a)</td><td>129</td><td>129</td></tr>
      <tr><td>Total capitalization</td><td>$</td><td>8,509</td></tr>
    </table>
    """

    def test_parses_two_data_columns(self):
        t = parse_table(_table(self.HTML))
        self.assertIsNotNone(t)
        self.assertEqual(len(t.headers), 2)

    def test_caption_detected(self):
        t = parse_table(_table(self.HTML))
        self.assertEqual(t.caption, "CAPITALIZATION")

    def test_row_values_correctly_attributed(self):
        t = parse_table(_table(self.HTML))
        idx = t.raw_row_labels.index("Total debt(a)")
        self.assertEqual(t.rows[idx], ["129", "129"])


class TestCurrencySymbolMerge(unittest.TestCase):
    """Regression: '$' rendered in its own <td>, confirmed live on every
    GEV/INBX/GRAL financial table — must merge into the following numeric cell."""

    HTML = """
    <table>
      <tr><td></td><td>Historical</td><td></td><td>Pro Forma</td></tr>
      <tr><td>Cash</td><td>$</td><td>1,551</td><td>$</td><td>3,597</td></tr>
    </table>
    """

    def test_dollar_sign_merges_into_number(self):
        t = parse_table(_table(self.HTML))
        self.assertIsNotNone(t)
        row = t.rows[t.raw_row_labels.index("Cash")]
        self.assertIn("$1,551", row)


class TestNegativeNumberSuffixMerge(unittest.TestCase):
    """Regression: accounting-style negatives split ')' into its own <td>
    (confirmed live on GEV's 'Accumulated other comprehensive income' row).
    Every data row must consistently place the ')' in the same column
    position for the all-rows detection to recognize it as a suffix column
    — matching the real filing's markup, where every row in a table has the
    same number of <td> cells regardless of whether that row's values are
    negative."""

    HTML = """
    <table>
      <tr><td></td><td>Historical</td><td></td><td>Pro Forma</td></tr>
      <tr><td>AOCI</td><td>(635</td><td>)</td><td>(635</td><td>)</td></tr>
      <tr><td>Total equity</td><td>7,416</td><td></td><td>8,004</td><td></td></tr>
    </table>
    """

    def test_close_paren_merges_into_previous_cell(self):
        t = parse_table(_table(self.HTML))
        self.assertIsNotNone(t)
        row = t.rows[t.raw_row_labels.index("AOCI")]
        self.assertIn("(635)", row)
        self.assertNotIn(")", row)  # no stray lone paren left as its own value

    def test_positive_row_unaffected_by_merge(self):
        t = parse_table(_table(self.HTML))
        row = t.rows[t.raw_row_labels.index("Total equity")]
        self.assertEqual(row, ["7,416", "8,004"])


class TestPercentSuffixMerge(unittest.TestCase):
    """Regression: 'Change %' columns split '%' (and '%)' for negatives)
    into their own <td>, confirmed live on GRAL's 'Results of Operations' table."""

    HTML = """
    <table>
      <tr><td></td><td>Change $</td><td>Change %</td></tr>
      <tr><td>Revenue</td><td>$8,090</td><td>53</td><td>%</td></tr>
      <tr><td>Costs</td><td>(123)</td><td>(49</td><td>%)</td></tr>
    </table>
    """

    def test_percent_sign_merges(self):
        t = parse_table(_table(self.HTML))
        self.assertIsNotNone(t)
        row = t.rows[t.raw_row_labels.index("Revenue")]
        self.assertIn("53%", row)

    def test_negative_percent_merges(self):
        t = parse_table(_table(self.HTML))
        row = t.rows[t.raw_row_labels.index("Costs")]
        self.assertIn("(49%)", row)


class TestSuccessorPredecessorSplit(unittest.TestCase):
    """
    Regression for the core bug found live against GRAL's real Form 10:
    a 6-period income statement with colspan=2 group headers over pairs of
    ($, number) columns must NOT collapse into a single ambiguous sequence
    of numbers with no per-column attribution.
    """

    HTML = """
    <table>
      <tr>
        <td></td>
        <td colspan="2" align="center">(Successor)</td>
        <td colspan="2" align="center">(Successor)</td>
        <td colspan="2" align="center">(Predecessor)</td>
      </tr>
      <tr>
        <td></td>
        <td colspan="2" align="center">Year Ended December 31, 2023</td>
        <td colspan="2" align="center">Year Ended January 1, 2023</td>
        <td colspan="2" align="center">January 1 to August 18, 2021</td>
      </tr>
      <tr><td></td><td colspan="6" align="center">(unaudited)</td></tr>
      <tr><td>Total revenue</td><td>$</td><td>93,105</td><td>$</td><td>55,550</td><td>$</td><td>2,179</td></tr>
    </table>
    """

    def test_three_period_columns_preserved(self):
        t = parse_table(_table(self.HTML))
        self.assertIsNotNone(t)
        self.assertEqual(len(t.headers), 3)

    def test_values_correctly_attributed_to_their_period(self):
        t = parse_table(_table(self.HTML))
        row = t.rows[t.raw_row_labels.index("Total revenue")]
        by_header = dict(zip(t.headers, row))
        # the January 1, 2023 period specifically must show 55,550 — this
        # is the exact value that must match XBRL's independently-resolved
        # last_year_sales for GRAL ($55,550,000)
        jan_col = next(h for h in t.headers if "January 1, 2023" in h)
        self.assertEqual(by_header[jan_col], "$55,550")

    def test_predecessor_label_only_on_its_own_column(self):
        t = parse_table(_table(self.HTML))
        predecessor_cols = [h for h in t.headers if "(Predecessor)" in h]
        self.assertEqual(len(predecessor_cols), 1)

    def test_unaudited_annotation_row_not_treated_as_data(self):
        t = parse_table(_table(self.HTML))
        self.assertNotIn("(unaudited)", t.raw_row_labels)


class TestUnlabeledColumnFlag(unittest.TestCase):
    """Tables with genuinely ambiguous leftover columns (footnote refs,
    e.g.) must be flagged rather than silently presented as clean. Header
    row must declare the same number of columns as the data rows, or a
    trailing footnote-reference column with no header text at all — same
    shape as the real Pro Forma adjustment tables found live in GEV's Form 10."""

    HTML = """
    <table>
      <tr><td></td><td>Historical</td><td>Adjustments</td><td></td></tr>
      <tr><td>Cash</td><td>1,551</td><td>2,046</td><td>(a), (b)</td></tr>
      <tr><td>Receivables</td><td>7,409</td><td>76</td><td>(c)</td></tr>
    </table>
    """

    def test_flags_unlabeled_columns_when_present(self):
        t = parse_table(_table(self.HTML))
        self.assertIsNotNone(t)
        self.assertTrue(t.has_unlabeled_columns)

    def test_clean_table_not_flagged(self):
        t = parse_table(_table(TestSimpleTwoColumnTable.HTML))
        self.assertFalse(t.has_unlabeled_columns)


class TestToLlmTextOmitsUnlabeledColumns(unittest.TestCase):
    HTML = TestUnlabeledColumnFlag.HTML

    def test_default_omits_unlabeled_fragments(self):
        t = parse_table(_table(self.HTML))
        text = t.to_llm_text()
        self.assertNotIn("Column ", text)

    def test_include_flag_shows_them(self):
        t = parse_table(_table(self.HTML))
        text = t.to_llm_text(include_unlabeled_columns=True)
        self.assertIn("Column ", text)


class TestZeroWidthSpaceHandling(unittest.TestCase):
    """
    Regression for a bug found live against Inhibrx's real Form 10: some
    filers' HTML templates use U+200B (zero-width space) as spacer content
    instead of &nbsp;. str.strip() does NOT remove U+200B, so a "blank"
    spacer cell survived as a non-empty '​' string, defeating both
    spacer-column collapse and header dedup — the entire pro forma balance
    sheet table's headers ran on into one unreadable 900+ character string.
    """

    HTML = (
        "<table>"
        "<tr><td>​</td><td>Historical​</td><td>​</td><td>​</td><td>Pro Forma​</td></tr>"
        "<tr><td>Cash</td><td>​</td><td>$</td><td>277,924</td><td>277,924</td></tr>"
        "</table>"
    )

    def test_zero_width_spacer_column_collapses(self):
        t = parse_table(_table(self.HTML))
        self.assertIsNotNone(t)
        # only 2 real data columns (Historical, Pro Forma) should remain
        self.assertEqual(len(t.headers), 2)

    def test_zero_width_space_stripped_from_header_text(self):
        t = parse_table(_table(self.HTML))
        for h in t.headers:
            self.assertNotIn("​", h)


class TestHeaderRowWithoutColspan(unittest.TestCase):
    """
    Regression: header detection previously required at least one cell to
    have colspan > 1, which broke perfectly ordinary tables where every
    <td> is a plain single-column cell (no grouping at all). Confirmed
    live this rejected the header row entirely, returning None for tables
    that should parse fine.
    """

    HTML = """
    <table>
      <tr><td></td><td>Historical</td><td>Pro Forma</td></tr>
      <tr><td>Cash</td><td>$</td><td>1,551</td><td>$</td><td>3,597</td></tr>
    </table>
    """

    def test_parses_without_any_colspan_present(self):
        t = parse_table(_table(self.HTML))
        self.assertIsNotNone(t)
        self.assertEqual(len(t.headers), 2)


class TestPerRowCurrencyMerge(unittest.TestCase):
    """
    Regression: a whole-column merge rule ("this column is $-only or blank
    in EVERY row") breaks on standard accounting formatting where '$' only
    appears on the first/subtotal row of a block, and the SAME column has
    a real plain number in other rows (confirmed live on GRAL's 10-K —
    "Gross loss" row has '$' in a column where "Amortization of intangible
    assets" two rows below has the literal number '133,889'). The merge
    must work per-row/per-cell, not assume column-wide consistency.
    """

    HTML = """
    <table>
      <tr><td></td><td colspan="2">2024</td><td colspan="2">2023</td></tr>
      <tr><td>Gross loss</td><td>$</td><td>(78,022)</td><td>$</td><td>(95,611)</td></tr>
      <tr><td>Amortization</td><td>133,889</td><td></td><td>133,889</td><td></td></tr>
    </table>
    """

    def test_dollar_only_merges_on_rows_that_have_it(self):
        t = parse_table(_table(self.HTML))
        self.assertIsNotNone(t)
        row = t.rows[t.raw_row_labels.index("Gross loss")]
        self.assertIn("$(78,022)", row)

    def test_plain_number_row_unaffected_in_same_column(self):
        t = parse_table(_table(self.HTML))
        row = t.rows[t.raw_row_labels.index("Amortization")]
        self.assertIn("133,889", row)
        self.assertNotIn("$133,889", row)  # no '$' to merge on this row — must not fabricate one


class TestGroupingHeaderFollowedByDistinctLabels(unittest.TestCase):
    """
    Regression: a wide grouping header ("Year Ended" spanning one colspan
    block) is textually identical to a throwaway annotation like
    "(unaudited)" by every signal internal to the row alone (confirmed
    live on GRAL's 10-K). The row below disambiguates: if it introduces
    multiple distinct labels (period dates), the row above is a real
    header, not an annotation to discard.
    """

    HTML = """
    <table>
      <tr><td></td><td colspan="2" align="center">Year Ended</td><td colspan="2" align="center">Year Ended</td></tr>
      <tr><td></td><td colspan="2" align="center">December 31, 2024</td><td colspan="2" align="center">December 31, 2023</td></tr>
      <tr><td>Revenue</td><td>$</td><td>100</td><td>$</td><td>90</td></tr>
    </table>
    """

    def test_grouping_header_row_retained_not_treated_as_annotation(self):
        t = parse_table(_table(self.HTML))
        self.assertIsNotNone(t)
        self.assertTrue(any("Year Ended" in h for h in t.headers))

    def test_period_dates_merged_with_grouping_label(self):
        t = parse_table(_table(self.HTML))
        matching = [h for h in t.headers if "December 31, 2024" in h]
        self.assertTrue(matching)


class TestSparseTablesRejected(unittest.TestCase):
    def test_single_cell_layout_table_returns_none(self):
        t = parse_table(_table("<table><tr><td>just one cell</td></tr></table>"))
        self.assertIsNone(t)

    def test_table_with_no_header_row_returns_none(self):
        # every row looks like data (all-numeric) — no identifiable header
        html = "<table><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>"
        t = parse_table(_table(html))
        self.assertIsNone(t)


class TestExtractTables(unittest.TestCase):
    def test_extracts_multiple_tables_from_document(self):
        html = f"""
        <html><body>
        {TestSimpleTwoColumnTable.HTML}
        <p>some prose in between</p>
        {TestPercentSuffixMerge.HTML}
        </body></html>
        """
        tables = extract_tables(html)
        self.assertEqual(len(tables), 2)

    def test_min_rows_filters_sparse_tables(self):
        html = "<table><tr><td>a</td><td>b</td></tr></table>"  # 1 row only
        tables = extract_tables(html, min_rows=2)
        self.assertEqual(tables, [])


if __name__ == "__main__":
    unittest.main()

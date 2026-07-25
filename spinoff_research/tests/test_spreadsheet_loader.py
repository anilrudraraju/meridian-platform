import unittest
from pathlib import Path

from spinoff_research.spreadsheet_loader import load_workbook, _map_column

_FIXTURE = Path(__file__).parent / "fixtures" / "sample_spinoff_workbook.xlsx"


class TestSpreadsheetLoader(unittest.TestCase):
    def test_loads_all_non_blank_rows(self):
        result = load_workbook(_FIXTURE)
        # 4 rows in the fixture, one is fully blank -> 3 loaded
        self.assertEqual(len(result.rows), 3)

    def test_skips_fully_blank_rows(self):
        result = load_workbook(_FIXTURE)
        source_row_numbers = [r.source_row_number for r in result.rows]
        # sequential 1,2,3 despite the blank row in between in the sheet
        self.assertEqual(source_row_numbers, [1, 2, 3])

    def test_maps_known_columns_case_insensitively(self):
        # fixture header is "Parent Company " (trailing space, title case);
        # dictionary's canonical name is "Parent company" (lowercase c)
        self.assertEqual(_map_column("Parent Company "), "parent_company_name")
        self.assertEqual(_map_column("parent company"), "parent_company_name")
        self.assertEqual(_map_column("PARENT COMPANY"), "parent_company_name")

    def test_unmapped_column_returns_none(self):
        self.assertIsNone(_map_column("Some Totally Unrecognized Column"))

    def test_reads_literal_values(self):
        result = load_workbook(_FIXTURE)
        row = result.rows[0]
        cell = row.get("parent_company_name")
        self.assertEqual(cell.raw_value, "Parent A")
        self.assertFalse(cell.is_formula)

    def test_detects_formula_cells(self):
        result = load_workbook(_FIXTURE)
        row = result.rows[0]
        cell = row.get("parent_market_cap")
        self.assertTrue(cell.is_formula)
        self.assertTrue(cell.formula_text.startswith("="))

    def test_formula_without_cached_result_is_flagged_not_blank(self):
        """
        openpyxl only returns a formula's cached result if the workbook was
        last saved by an app that evaluated it (Excel/LibreOffice). Files
        written by openpyxl itself (like this test fixture) have formulas
        with no cached value — that must surface as has_stale_formula_cache,
        not get silently miscounted as a blank/not-found cell.
        """
        result = load_workbook(_FIXTURE)
        row = result.rows[0]
        cell = row.get("parent_market_cap")
        self.assertTrue(cell.is_formula)
        self.assertTrue(cell.has_stale_formula_cache)
        self.assertFalse(cell.is_blank)  # distinct from a genuinely blank cell

    def test_detects_blank_cells_distinctly_from_zero_or_string(self):
        result = load_workbook(_FIXTURE)
        blank_row = result.rows[1]  # "Test Co B" row — mostly blank fields
        announcement_cell = blank_row.get("announcement_date")
        self.assertTrue(announcement_cell.is_blank)
        self.assertIsNone(announcement_cell.raw_value)

    def test_missing_field_not_confused_with_blank_cell(self):
        """
        A field_key with no matching column at all (missing_fields) must be
        distinguishable from a field_key that has a column but a blank value
        in a given row (is_blank=True) — these are different failure modes.
        """
        result = load_workbook(_FIXTURE)
        self.assertIn("parent_retained_ownership", result.missing_fields)
        row = result.rows[0]
        self.assertIsNone(row.get("parent_retained_ownership"))  # no cell at all, not a blank one

    def test_no_unmapped_columns_in_fixture(self):
        result = load_workbook(_FIXTURE)
        self.assertEqual(result.unmapped_columns, [])

    def test_does_not_mutate_source_file(self):
        import hashlib
        before = hashlib.sha256(_FIXTURE.read_bytes()).hexdigest()
        load_workbook(_FIXTURE)
        after = hashlib.sha256(_FIXTURE.read_bytes()).hexdigest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

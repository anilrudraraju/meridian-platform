import tempfile
import unittest
from pathlib import Path

from spinoff_research.db import init_db
from spinoff_research.extraction import (
    ExtractedFieldValue,
    SourceCitation,
    persist_field_value,
    get_current_field_value,
    get_all_current_field_values,
)
from spinoff_research.status import FieldStatus


class TestExtraction(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.conn = init_db(Path(self._tmpdir.name) / "test.db")
        self.conn.execute("INSERT INTO companies (company_id, name, ticker) VALUES (1, 'Parent', 'P')")
        self.conn.execute("INSERT INTO companies (company_id, name, ticker) VALUES (2, 'Spinco', 'S')")
        self.conn.execute(
            "INSERT INTO spinoff_transactions (transaction_id, parent_company_id, spinoff_company_id) VALUES (1, 1, 2)"
        )
        self.conn.commit()

    def tearDown(self):
        self.conn.close()
        self._tmpdir.cleanup()

    def _simple_result(self, field_key="test_field", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE):
        return ExtractedFieldValue(
            field_key=field_key, extraction_method="xbrl", status=status,
            raw_value="42", numeric_value=42.0, confidence=0.9,
        )

    def test_persist_writes_run_and_value_rows(self):
        fv_id = persist_field_value(self.conn, 1, self._simple_result())
        run_count = self.conn.execute("SELECT COUNT(*) AS c FROM field_extraction_runs").fetchone()["c"]
        value_count = self.conn.execute("SELECT COUNT(*) AS c FROM field_values").fetchone()["c"]
        self.assertEqual(run_count, 1)
        self.assertEqual(value_count, 1)
        self.assertIsNotNone(fv_id)

    def test_persist_links_run_id_to_value(self):
        persist_field_value(self.conn, 1, self._simple_result())
        row = self.conn.execute("SELECT * FROM field_values").fetchone()
        run = self.conn.execute("SELECT * FROM field_extraction_runs WHERE run_id = ?", (row["run_id"],)).fetchone()
        self.assertIsNotNone(run)
        self.assertEqual(run["field_key"], "test_field")

    def test_persist_writes_sources(self):
        result = self._simple_result()
        result.sources = [
            SourceCitation(supporting_excerpt="excerpt A", reasoning_summary="reason A"),
            SourceCitation(supporting_excerpt="excerpt B", reasoning_summary="reason B"),
        ]
        fv_id = persist_field_value(self.conn, 1, result)
        sources = self.conn.execute("SELECT * FROM field_sources WHERE field_value_id = ?", (fv_id,)).fetchall()
        self.assertEqual(len(sources), 2)
        self.assertEqual({s["supporting_excerpt"] for s in sources}, {"excerpt A", "excerpt B"})

    def test_persist_with_no_sources_writes_zero_source_rows(self):
        fv_id = persist_field_value(self.conn, 1, self._simple_result())
        sources = self.conn.execute("SELECT * FROM field_sources WHERE field_value_id = ?", (fv_id,)).fetchall()
        self.assertEqual(sources, [])

    def test_status_persisted_as_string_value(self):
        fv_id = persist_field_value(self.conn, 1, self._simple_result(status=FieldStatus.NOT_FOUND))
        row = self.conn.execute("SELECT status FROM field_values WHERE field_value_id = ?", (fv_id,)).fetchone()
        self.assertEqual(row["status"], "not_found")

    def test_get_current_field_value_returns_latest(self):
        persist_field_value(self.conn, 1, self._simple_result())
        current = get_current_field_value(self.conn, 1, "test_field")
        self.assertIsNotNone(current)
        self.assertEqual(current["numeric_value"], 42.0)

    def test_get_current_field_value_none_when_not_extracted(self):
        self.assertIsNone(get_current_field_value(self.conn, 1, "never_extracted"))

    def test_get_current_field_value_scoped_to_transaction(self):
        self.conn.execute("INSERT INTO companies (company_id, name, ticker) VALUES (3, 'Parent2', 'P2')")
        self.conn.execute("INSERT INTO companies (company_id, name, ticker) VALUES (4, 'Spinco2', 'S2')")
        self.conn.execute(
            "INSERT INTO spinoff_transactions (transaction_id, parent_company_id, spinoff_company_id) VALUES (2, 3, 4)"
        )
        self.conn.commit()
        persist_field_value(self.conn, 1, self._simple_result())
        self.assertIsNone(get_current_field_value(self.conn, 2, "test_field"))

    def test_calling_twice_creates_two_rows_not_an_update(self):
        """persist_field_value never supersedes — that's a review-workflow concern."""
        persist_field_value(self.conn, 1, self._simple_result())
        persist_field_value(self.conn, 1, self._simple_result(status=FieldStatus.EXTRACTED_UNCERTAIN))
        count = self.conn.execute("SELECT COUNT(*) AS c FROM field_values WHERE field_key = 'test_field'").fetchone()["c"]
        self.assertEqual(count, 2)

    def test_get_all_current_field_values_one_row_per_field(self):
        persist_field_value(self.conn, 1, self._simple_result(field_key="field_a"))
        persist_field_value(self.conn, 1, self._simple_result(field_key="field_b"))
        rows = get_all_current_field_values(self.conn, 1)
        self.assertEqual({r["field_key"] for r in rows}, {"field_a", "field_b"})


if __name__ == "__main__":
    unittest.main()

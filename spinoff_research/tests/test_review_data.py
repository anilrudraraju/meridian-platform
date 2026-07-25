import tempfile
import unittest
from pathlib import Path

from spinoff_research.db import init_db
from spinoff_research.extraction import ExtractedFieldValue, SourceCitation, persist_field_value
from spinoff_research.review_data import load_review_rows, missing_deterministic_fields, DETERMINISTIC_FIELD_KEYS
from spinoff_research.status import FieldStatus


class TestReviewData(unittest.TestCase):
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

    def test_load_review_rows_resolves_display_name(self):
        result = ExtractedFieldValue(field_key="form_10_availability", extraction_method="filing_metadata", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE, raw_value="True")
        persist_field_value(self.conn, 1, result)
        rows = load_review_rows(self.conn, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].display_name, "Form 10 availability")

    def test_load_review_rows_joins_sources(self):
        result = ExtractedFieldValue(
            field_key="spinoff_debt", extraction_method="xbrl", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE,
            raw_value="129000000",
            sources=[SourceCitation(supporting_excerpt="us-gaap:LongTermDebt = 129000000", reasoning_summary="single candidate")],
        )
        persist_field_value(self.conn, 1, result)
        rows = load_review_rows(self.conn, 1)
        self.assertEqual(len(rows[0].sources), 1)
        self.assertEqual(rows[0].sources[0].supporting_excerpt, "us-gaap:LongTermDebt = 129000000")

    def test_load_review_rows_scoped_to_transaction(self):
        self.conn.execute("INSERT INTO companies (company_id, name, ticker) VALUES (3, 'P2', 'P2')")
        self.conn.execute("INSERT INTO companies (company_id, name, ticker) VALUES (4, 'S2', 'S2')")
        self.conn.execute(
            "INSERT INTO spinoff_transactions (transaction_id, parent_company_id, spinoff_company_id) VALUES (2, 3, 4)"
        )
        self.conn.commit()
        persist_field_value(self.conn, 1, ExtractedFieldValue(field_key="form_10_availability", extraction_method="filing_metadata", status=FieldStatus.NOT_FOUND))
        persist_field_value(self.conn, 2, ExtractedFieldValue(field_key="form_10_availability", extraction_method="filing_metadata", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE))
        rows_txn1 = load_review_rows(self.conn, 1)
        self.assertEqual(len(rows_txn1), 1)
        self.assertEqual(rows_txn1[0].status, "not_found")

    def test_unknown_field_key_falls_back_to_raw_key_as_display_name(self):
        persist_field_value(self.conn, 1, ExtractedFieldValue(field_key="not_a_real_field", extraction_method="xbrl", status=FieldStatus.NOT_FOUND))
        rows = load_review_rows(self.conn, 1)
        self.assertEqual(rows[0].display_name, "not_a_real_field")

    def test_missing_deterministic_fields_when_none_attempted(self):
        missing = missing_deterministic_fields([])
        self.assertEqual(set(missing), DETERMINISTIC_FIELD_KEYS)

    def test_missing_deterministic_fields_when_all_attempted(self):
        for key in DETERMINISTIC_FIELD_KEYS:
            persist_field_value(self.conn, 1, ExtractedFieldValue(field_key=key, extraction_method="xbrl", status=FieldStatus.NOT_FOUND))
        rows = load_review_rows(self.conn, 1)
        self.assertEqual(missing_deterministic_fields(rows), [])


if __name__ == "__main__":
    unittest.main()

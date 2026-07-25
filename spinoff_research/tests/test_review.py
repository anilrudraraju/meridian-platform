import tempfile
import unittest
from pathlib import Path

from spinoff_research.db import init_db
from spinoff_research.extraction import ExtractedFieldValue, persist_field_value, get_current_field_value
from spinoff_research.review import record_review, get_review_history
from spinoff_research.status import FieldStatus, ReviewAction


class TestReview(unittest.TestCase):
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

    def _persist(self, field_key="test_field", status=FieldStatus.EXTRACTED_UNCERTAIN, raw_value="100"):
        result = ExtractedFieldValue(field_key=field_key, extraction_method="xbrl", status=status, raw_value=raw_value)
        return persist_field_value(self.conn, 1, result)

    def test_approve_updates_status_in_place(self):
        fv_id = self._persist()
        record_review(self.conn, fv_id, ReviewAction.APPROVE, reviewed_by="Rich")
        row = self.conn.execute("SELECT * FROM field_values WHERE field_value_id = ?", (fv_id,)).fetchone()
        self.assertEqual(row["status"], "approved")
        self.assertEqual(row["raw_value"], "100")  # unchanged

    def test_approve_does_not_create_new_row(self):
        fv_id = self._persist()
        record_review(self.conn, fv_id, ReviewAction.APPROVE)
        count = self.conn.execute("SELECT COUNT(*) AS c FROM field_values").fetchone()["c"]
        self.assertEqual(count, 1)

    def test_reject_updates_status_in_place(self):
        fv_id = self._persist()
        record_review(self.conn, fv_id, ReviewAction.REJECT, reviewer_notes="wrong source")
        row = self.conn.execute("SELECT * FROM field_values WHERE field_value_id = ?", (fv_id,)).fetchone()
        self.assertEqual(row["status"], "rejected")

    def test_edit_preserves_original_and_creates_new_row(self):
        """Core requirement: a correction must never overwrite the original extraction."""
        fv_id = self._persist(raw_value="wrong_value")
        record_review(self.conn, fv_id, ReviewAction.EDIT, corrected_value="right_value", reviewed_by="Rich")

        original = self.conn.execute("SELECT * FROM field_values WHERE field_value_id = ?", (fv_id,)).fetchone()
        self.assertEqual(original["raw_value"], "wrong_value")  # untouched
        self.assertEqual(original["is_original_extraction"], 0)
        self.assertIsNotNone(original["superseded_by_field_value_id"])

        current = get_current_field_value(self.conn, 1, "test_field")
        self.assertEqual(current["raw_value"], "right_value")
        self.assertEqual(current["status"], "manually_entered")

    def test_edit_links_original_to_correction(self):
        fv_id = self._persist()
        record_review(self.conn, fv_id, ReviewAction.EDIT, corrected_value="corrected")
        original = self.conn.execute("SELECT * FROM field_values WHERE field_value_id = ?", (fv_id,)).fetchone()
        correction = self.conn.execute(
            "SELECT * FROM field_values WHERE field_value_id = ?", (original["superseded_by_field_value_id"],)
        ).fetchone()
        self.assertEqual(correction["raw_value"], "corrected")

    def test_mark_not_found_creates_not_found_row(self):
        fv_id = self._persist(raw_value="questionable")
        record_review(self.conn, fv_id, ReviewAction.MARK_NOT_FOUND, reviewer_notes="couldn't verify")
        current = get_current_field_value(self.conn, 1, "test_field")
        self.assertEqual(current["status"], "not_found")
        self.assertIsNone(current["raw_value"])

    def test_unknown_field_value_id_raises(self):
        with self.assertRaises(ValueError):
            record_review(self.conn, 99999, ReviewAction.APPROVE)

    def test_review_history_recorded(self):
        fv_id = self._persist()
        record_review(self.conn, fv_id, ReviewAction.APPROVE, reviewer_notes="looks right", reviewed_by="Rich")
        history = get_review_history(self.conn, fv_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["action"], "approve")
        self.assertEqual(history[0]["reviewed_by"], "Rich")

    def test_multiple_reviews_all_preserved_in_history(self):
        """Audit trail requirement: never delete or overwrite a prior review row."""
        fv_id = self._persist()
        record_review(self.conn, fv_id, ReviewAction.REJECT, reviewer_notes="first pass")
        record_review(self.conn, fv_id, ReviewAction.APPROVE, reviewer_notes="changed my mind")
        history = get_review_history(self.conn, fv_id)
        self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()

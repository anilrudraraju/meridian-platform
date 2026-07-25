import tempfile
import unittest
from pathlib import Path

from spinoff_research.db import init_db


class TestDbSchema(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "test.db"

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_init_db_creates_all_tables(self):
        conn = init_db(self.db_path)
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        expected = {
            "companies", "spinoff_transactions", "documents", "document_sections",
            "field_extraction_runs", "field_values", "field_sources",
            "field_reviews", "scheduled_field_requirements",
        }
        self.assertTrue(expected.issubset(tables), f"missing: {expected - tables}")
        conn.close()

    def test_init_db_is_idempotent(self):
        init_db(self.db_path).close()
        conn = init_db(self.db_path)  # second call should not raise
        count = conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
        self.assertEqual(count, 0)
        conn.close()

    def test_foreign_keys_enforced(self):
        conn = init_db(self.db_path)
        with self.assertRaises(Exception):
            conn.execute(
                "INSERT INTO spinoff_transactions (parent_company_id, spinoff_company_id) VALUES (999, 998)"
            )
            conn.commit()
        conn.close()

    def test_original_extraction_preserved_after_supersede(self):
        """
        A reviewed/corrected field_value must not overwrite the original —
        it gets a new row, and the old row's is_original_extraction flag
        and superseded_by pointer make the audit trail queryable.
        """
        conn = init_db(self.db_path)
        conn.execute(
            "INSERT INTO companies (company_id, name, ticker) VALUES (1, 'Parent X', 'PX')"
        )
        conn.execute(
            "INSERT INTO companies (company_id, name, ticker) VALUES (2, 'Spinco X', 'SX')"
        )
        conn.execute(
            "INSERT INTO spinoff_transactions (transaction_id, parent_company_id, spinoff_company_id) "
            "VALUES (1, 1, 2)"
        )
        conn.execute(
            "INSERT INTO field_values (field_value_id, transaction_id, field_key, raw_value, status) "
            "VALUES (1, 1, 'ceo_came_from_parent', 'Yes', 'extracted_high_confidence')"
        )
        conn.execute(
            "INSERT INTO field_values (field_value_id, transaction_id, field_key, raw_value, status, is_original_extraction) "
            "VALUES (2, 1, 'ceo_came_from_parent', 'No', 'approved', 0)"
        )
        conn.execute("UPDATE field_values SET superseded_by_field_value_id = 2 WHERE field_value_id = 1")
        conn.commit()

        original = conn.execute(
            "SELECT * FROM field_values WHERE field_value_id = 1"
        ).fetchone()
        self.assertEqual(original["raw_value"], "Yes")
        self.assertEqual(original["superseded_by_field_value_id"], 2)
        self.assertEqual(original["is_original_extraction"], 1)
        conn.close()


if __name__ == "__main__":
    unittest.main()

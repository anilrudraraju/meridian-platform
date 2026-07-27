import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from spinoff_research.db import init_db
from spinoff_research.models import Company, SpinoffTransaction
from spinoff_research.repository import get_or_create_transaction
from spinoff_research.status import FieldStatus


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.conn = init_db(Path(self._tmpdir.name) / "test.db")

    def tearDown(self):
        self.conn.close()
        self._tmpdir.cleanup()

    def test_raises_if_transaction_not_persisted(self):
        from spinoff_research.orchestrator import run_deterministic_extraction
        unsaved = SpinoffTransaction(
            parent=Company(name="Parent", ticker="P"),
            spinoff=Company(name="Spinco", ticker="S"),
        )
        with self.assertRaises(ValueError):
            run_deterministic_extraction(self.conn, unsaved)

    def test_skips_fields_needing_missing_cik_and_records_error(self):
        """No cik/spinoff_date at all — most fields should be skipped
        cleanly with a logged reason, not crash."""
        from spinoff_research.orchestrator import run_deterministic_extraction
        tx = SpinoffTransaction(parent=Company(name="Parent"), spinoff=Company(name="Spinco"))
        saved = get_or_create_transaction(self.conn, tx)
        summary = run_deterministic_extraction(self.conn, saved)
        self.assertEqual(summary.field_count, 0)
        self.assertGreater(len(summary.errors), 0)

    @patch("spinoff_research.orchestrator.extract_form10_bool_field")
    @patch("spinoff_research.orchestrator.extract_share_price")
    @patch("spinoff_research.orchestrator.extract_sector_industry")
    @patch("spinoff_research.orchestrator.extract_insider_buying_fields")
    @patch("spinoff_research.orchestrator.extract_xbrl_field")
    @patch("spinoff_research.orchestrator.extract_form_10_availability")
    @patch("spinoff_research.orchestrator.extract_company_identity")
    @patch("spinoff_research.orchestrator.resolve_cik")
    def test_full_run_persists_all_results(
        self, mock_resolve_cik, mock_identity, mock_form10, mock_xbrl, mock_form4, mock_sector, mock_price, mock_ai_bool,
    ):
        from spinoff_research.extraction import ExtractedFieldValue, get_all_current_field_values
        from spinoff_research.orchestrator import run_deterministic_extraction

        mock_resolve_cik.return_value = {"cik": "0001996810", "name": "Test Co"}
        mock_identity.return_value = [
            ExtractedFieldValue(field_key="x_name", extraction_method="filing_metadata", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE),
        ]
        mock_form10.return_value = ExtractedFieldValue(field_key="form_10_availability", extraction_method="filing_metadata", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE)
        mock_xbrl.return_value = ExtractedFieldValue(field_key="xbrl_field", extraction_method="xbrl", status=FieldStatus.EXTRACTED_UNCERTAIN)
        mock_form4.return_value = [
            ExtractedFieldValue(field_key="insider_buying_within_3mo", extraction_method="form4_aggregation", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE),
        ]
        mock_sector.return_value = [
            ExtractedFieldValue(field_key="sector_field", extraction_method="market_data", status=FieldStatus.EXTRACTED_UNCERTAIN),
        ]
        mock_price.return_value = ExtractedFieldValue(field_key="price_field", extraction_method="market_data", status=FieldStatus.EXTRACTED_UNCERTAIN)
        mock_ai_bool.side_effect = lambda conn, txn_id, cik, name, field_key: ExtractedFieldValue(
            field_key=field_key, extraction_method="ai_assisted", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE,
        )

        tx = SpinoffTransaction(
            parent=Company(name="Parent", ticker="P", cik="0000000001"),
            spinoff=Company(name="Spinco", ticker="S", cik="0000000002"),
            announcement_date="2024-01-01", spinoff_date="2024-04-01",
        )
        saved = get_or_create_transaction(self.conn, tx)
        summary = run_deterministic_extraction(self.conn, saved)

        self.assertEqual(summary.errors, [])
        persisted = get_all_current_field_values(self.conn, saved.transaction_id)
        self.assertGreater(len(persisted), 0)
        # every mocked extractor's output actually made it into the DB
        persisted_keys = {r["field_key"] for r in persisted}
        self.assertIn("form_10_availability", persisted_keys)
        self.assertIn("insider_buying_within_3mo", persisted_keys)
        self.assertIn("ceo_came_from_parent", persisted_keys)
        # all 6 bool fields ran — one call to the mock per field_key
        self.assertEqual(mock_ai_bool.call_count, 6)

    @patch("spinoff_research.orchestrator.extract_form10_bool_field")
    @patch("spinoff_research.orchestrator.extract_share_price")
    @patch("spinoff_research.orchestrator.extract_sector_industry")
    @patch("spinoff_research.orchestrator.extract_insider_buying_fields")
    @patch("spinoff_research.orchestrator.extract_xbrl_field")
    @patch("spinoff_research.orchestrator.extract_form_10_availability")
    @patch("spinoff_research.orchestrator.extract_company_identity")
    @patch("spinoff_research.orchestrator.resolve_cik")
    def test_skip_field_keys_excludes_approved_fields(
        self, mock_resolve_cik, mock_identity, mock_form10, mock_xbrl, mock_form4, mock_sector, mock_price, mock_ai_bool,
    ):
        """
        Regression for the Excel round-trip's core rule: fields Rich has
        approved must never be re-extracted or re-persisted on a later
        run, even though their extraction mechanism is still fully
        functional and would happily produce a value again.
        """
        from spinoff_research.extraction import ExtractedFieldValue, get_all_current_field_values
        from spinoff_research.orchestrator import run_deterministic_extraction

        mock_resolve_cik.return_value = {"cik": "0001996810", "name": "Test Co"}
        mock_identity.return_value = []
        mock_form10.return_value = ExtractedFieldValue(field_key="form_10_availability", extraction_method="filing_metadata", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE)
        mock_xbrl.return_value = ExtractedFieldValue(field_key="spinoff_debt", extraction_method="xbrl", status=FieldStatus.EXTRACTED_UNCERTAIN)
        mock_form4.return_value = [
            ExtractedFieldValue(field_key="insider_buying_within_3mo", extraction_method="form4_aggregation", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE),
            ExtractedFieldValue(field_key="cluster_insider_buying_within_3mo", extraction_method="form4_aggregation", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE),
            ExtractedFieldValue(field_key="insider_buyer_count", extraction_method="form4_aggregation", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE),
        ]
        mock_sector.return_value = []
        mock_price.return_value = ExtractedFieldValue(field_key="parent_share_price", extraction_method="market_data", status=FieldStatus.EXTRACTED_UNCERTAIN)
        mock_ai_bool.side_effect = lambda conn, txn_id, cik, name, field_key: ExtractedFieldValue(
            field_key=field_key, extraction_method="ai_assisted", status=FieldStatus.EXTRACTED_HIGH_CONFIDENCE,
        )

        tx = SpinoffTransaction(
            parent=Company(name="Parent", ticker="P", cik="0000000001"),
            spinoff=Company(name="Spinco", ticker="S", cik="0000000002"),
            announcement_date="2024-01-01", spinoff_date="2024-04-01",
        )
        saved = get_or_create_transaction(self.conn, tx)
        summary = run_deterministic_extraction(
            self.conn, saved,
            skip_field_keys={"form_10_availability", "insider_buying_within_3mo", "ceo_came_from_parent"},
        )

        persisted_keys = {r.field_key for r in summary.results}
        self.assertNotIn("form_10_availability", persisted_keys)
        self.assertNotIn("insider_buying_within_3mo", persisted_keys)
        self.assertNotIn("ceo_came_from_parent", persisted_keys)
        # the other two Form 4 fields from the SAME extractor call still
        # persist — skipping is per-field-key, not per-extraction-call
        self.assertIn("cluster_insider_buying_within_3mo", persisted_keys)
        self.assertIn("insider_buyer_count", persisted_keys)
        self.assertIn("spinoff_debt", persisted_keys)
        # the other 5 bool fields (not skipped) still ran
        self.assertIn("cfo_came_from_parent", persisted_keys)
        self.assertEqual(mock_ai_bool.call_count, 5)

        db_keys = {r["field_key"] for r in get_all_current_field_values(self.conn, saved.transaction_id)}
        self.assertNotIn("form_10_availability", db_keys)


if __name__ == "__main__":
    unittest.main()

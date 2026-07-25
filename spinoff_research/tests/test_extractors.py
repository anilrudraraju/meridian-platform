import unittest
from unittest.mock import patch, MagicMock

from spinoff_research.status import FieldStatus


class TestFilingMetadataExtractors(unittest.TestCase):
    @patch("spinoff_research.extractors.filing_metadata.discover_filings")
    def test_form_10_availability_found(self, mock_discover):
        mock_discover.return_value = [MagicMock(form_type="10-12B", filing_date="2024-02-15")]
        from spinoff_research.extractors.filing_metadata import extract_form_10_availability
        result = extract_form_10_availability("0001996810")
        self.assertEqual(result.status, FieldStatus.EXTRACTED_HIGH_CONFIDENCE)
        self.assertEqual(result.raw_value, "True")
        self.assertEqual(len(result.sources), 1)

    @patch("spinoff_research.extractors.filing_metadata.discover_filings")
    def test_form_10_availability_not_found(self, mock_discover):
        mock_discover.return_value = []
        from spinoff_research.extractors.filing_metadata import extract_form_10_availability
        result = extract_form_10_availability("0001996810")
        self.assertEqual(result.status, FieldStatus.NOT_FOUND)
        self.assertEqual(result.raw_value, "False")

    def test_company_identity_when_resolved(self):
        from spinoff_research.extractors.filing_metadata import extract_company_identity
        results = extract_company_identity("GEV", {"cik": "0001996810", "name": "GE Vernova Inc."}, "spinoff")
        self.assertEqual(len(results), 2)
        name_result = next(r for r in results if r.field_key == "spinoff_company_name")
        ticker_result = next(r for r in results if r.field_key == "spinoff_ticker")
        self.assertEqual(name_result.raw_value, "GE Vernova Inc.")
        self.assertEqual(ticker_result.raw_value, "GEV")
        self.assertEqual(name_result.status, FieldStatus.EXTRACTED_HIGH_CONFIDENCE)

    def test_company_identity_when_unresolved(self):
        from spinoff_research.extractors.filing_metadata import extract_company_identity
        results = extract_company_identity("ZZZZ", None, "parent")
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.status == FieldStatus.NOT_FOUND for r in results))


class TestXbrlExtractor(unittest.TestCase):
    @patch("spinoff_research.extractors.xbrl.resolve_field_for_transaction")
    @patch("spinoff_research.extractors.xbrl.fetch_company_facts")
    def test_maps_xbrl_result_to_extracted_field_value(self, mock_fetch, mock_resolve):
        mock_fetch.return_value = {"facts": {}}
        candidate = MagicMock(
            concept="us-gaap:CommonStockSharesOutstanding", value=274085523.0, unit="shares",
            period_start=None, period_end="2024-04-02", accession_number="0001-24-000001", form="10-Q",
        )
        mock_resolve.return_value = MagicMock(
            status="extracted_uncertain", selected=candidate, rejected_candidates=[],
            selection_reason="test reason", conflict_note=None,
        )
        from spinoff_research.extractors.xbrl import extract_xbrl_field
        result = extract_xbrl_field("spinoff_shares_outstanding", "0001996810", distribution_date="2024-04-02")
        self.assertEqual(result.status, FieldStatus.EXTRACTED_UNCERTAIN)
        self.assertEqual(result.numeric_value, 274085523.0)
        self.assertEqual(len(result.sources), 1)

    @patch("spinoff_research.extractors.xbrl.fetch_company_facts")
    def test_fetch_failure_returns_not_found(self, mock_fetch):
        from spinoff_research.xbrl_service import XbrlServiceError
        mock_fetch.side_effect = XbrlServiceError("network error")
        from spinoff_research.extractors.xbrl import extract_xbrl_field
        result = extract_xbrl_field("spinoff_debt", "0001996810", distribution_date="2024-04-02")
        self.assertEqual(result.status, FieldStatus.NOT_FOUND)

    @patch("spinoff_research.extractors.xbrl.resolve_field_for_transaction")
    @patch("spinoff_research.extractors.xbrl.fetch_company_facts")
    def test_conflicting_sources_no_selected_still_carries_evidence(self, mock_fetch, mock_resolve):
        mock_fetch.return_value = {"facts": {}}
        mock_resolve.return_value = MagicMock(
            status="conflicting_sources", selected=None,
            rejected_candidates=[MagicMock()], selection_reason="disagreement",
            conflict_note="tag_a=100; tag_b=200",
        )
        from spinoff_research.extractors.xbrl import extract_xbrl_field
        result = extract_xbrl_field("spinoff_debt", "0001996810", distribution_date="2024-04-02")
        self.assertEqual(result.status, FieldStatus.CONFLICTING_SOURCES)
        self.assertIsNone(result.numeric_value)
        self.assertEqual(len(result.sources), 1)


class TestForm4Extractor(unittest.TestCase):
    @patch("spinoff_research.extractors.form4._paced_get")
    @patch("spinoff_research.extractors.form4.discover_filings")
    def test_returns_three_fields(self, mock_discover, mock_get):
        mock_discover.return_value = []
        from spinoff_research.extractors.form4 import extract_insider_buying_fields
        results = extract_insider_buying_fields("0001996810", distribution_date="2024-04-02")
        keys = {r.field_key for r in results}
        self.assertEqual(keys, {"insider_buying_within_3mo", "cluster_insider_buying_within_3mo", "insider_buyer_count"})

    @patch("spinoff_research.extractors.form4._paced_get")
    @patch("spinoff_research.extractors.form4.discover_filings")
    def test_no_filings_means_confident_no_buying(self, mock_discover, mock_get):
        """A confidently-checked zero result is extracted_high_confidence,
        not not_found — the check happened and found nothing, which is
        different from failing to check."""
        mock_discover.return_value = []
        from spinoff_research.extractors.form4 import extract_insider_buying_fields
        results = extract_insider_buying_fields("0001996810", distribution_date="2024-04-02")
        buying = next(r for r in results if r.field_key == "insider_buying_within_3mo")
        self.assertEqual(buying.status, FieldStatus.EXTRACTED_HIGH_CONFIDENCE)
        self.assertEqual(buying.raw_value, "False")

    @patch("spinoff_research.extractors.form4._paced_get")
    @patch("spinoff_research.extractors.form4.discover_filings")
    def test_parse_failure_lowers_confidence_not_status(self, mock_discover, mock_get):
        from spinoff_research.sec_service import SecServiceError
        mock_discover.return_value = [MagicMock(raw_document_url="http://x", accession_number_dashed="acc-1")]
        mock_get.side_effect = SecServiceError("network blip")
        from spinoff_research.extractors.form4 import extract_insider_buying_fields
        results = extract_insider_buying_fields("0001996810", distribution_date="2024-04-02")
        buying = next(r for r in results if r.field_key == "insider_buying_within_3mo")
        self.assertLess(buying.confidence, 1.0)


class TestMarketDataExtractors(unittest.TestCase):
    def test_sector_industry_success(self):
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker_cls.return_value.info = {"sector": "Industrials", "industry": "Specialty Industrial Machinery"}
            from spinoff_research.extractors.market_data import extract_sector_industry
            results = extract_sector_industry("GEV", "spinoff")
            sector = next(r for r in results if r.field_key == "spinoff_sector")
            industry = next(r for r in results if r.field_key == "spinoff_industry")
            self.assertEqual(sector.raw_value, "Industrials")
            self.assertEqual(industry.raw_value, "Specialty Industrial Machinery")

    def test_sector_industry_always_uncertain_not_high_confidence(self):
        """Regression: even a successful yfinance lookup must never be
        high-confidence — its taxonomy doesn't always match Rich's own
        labels (confirmed live: ILMN)."""
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker_cls.return_value.info = {"sector": "Healthcare", "industry": "Diagnostics & Research"}
            from spinoff_research.extractors.market_data import extract_sector_industry
            results = extract_sector_industry("GRAL", "spinoff")
            self.assertTrue(all(r.status == FieldStatus.EXTRACTED_UNCERTAIN for r in results))

    def test_sector_industry_lookup_failure(self):
        with patch("yfinance.Ticker", side_effect=Exception("boom")):
            from spinoff_research.extractors.market_data import extract_sector_industry
            results = extract_sector_industry("BADTICKER", "parent")
            self.assertTrue(all(r.status == FieldStatus.NOT_FOUND for r in results))

    def test_share_price_always_uncertain(self):
        """Regression: Rich's own 3 pilot values don't follow one
        consistent date convention (GE matched day-before, ILMN matched
        day-of, SNY matched nothing) — this field can never be high
        confidence regardless of whether yfinance returns a clean value."""
        import pandas as pd
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_hist = pd.DataFrame({"Close": [140.00]}, index=pd.to_datetime(["2024-04-02"]))
            mock_ticker_cls.return_value.history.return_value = mock_hist
            from spinoff_research.extractors.market_data import extract_share_price
            result = extract_share_price("GEV", "spinoff_share_price", "2024-04-02")
            self.assertEqual(result.status, FieldStatus.EXTRACTED_UNCERTAIN)
            self.assertEqual(result.numeric_value, 140.00)

    def test_share_price_empty_history_is_not_found(self):
        import pandas as pd
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker_cls.return_value.history.return_value = pd.DataFrame({"Close": []})
            from spinoff_research.extractors.market_data import extract_share_price
            result = extract_share_price("GEV", "spinoff_share_price", "2024-04-02")
            self.assertEqual(result.status, FieldStatus.NOT_FOUND)


if __name__ == "__main__":
    unittest.main()

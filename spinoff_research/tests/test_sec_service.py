import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from spinoff_research.tests.fixtures.sample_submissions import (
    SAMPLE_TICKER_MAP_RAW,
    SAMPLE_SUBMISSIONS_DOMESTIC,
    SAMPLE_SUBMISSIONS_FPI,
)


def _mock_response(json_data):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_data
    return resp


class TestSecServiceBase(unittest.TestCase):
    """Base: isolates each test's on-disk cache dir and sets a valid User-Agent."""

    def setUp(self):
        os.environ["SEC_EDGAR_USER_AGENT"] = "TestAgent test@example.com"
        self._tmpdir = tempfile.TemporaryDirectory()
        self._cache_patch = patch(
            "spinoff_research.sec_service._CACHE_DIR", Path(self._tmpdir.name)
        )
        self._cache_map_patch = patch(
            "spinoff_research.sec_service._TICKER_MAP_CACHE",
            Path(self._tmpdir.name) / "company_tickers.json",
        )
        self._cache_patch.start()
        self._cache_map_patch.start()

    def tearDown(self):
        self._cache_patch.stop()
        self._cache_map_patch.stop()
        self._tmpdir.cleanup()


class TestResolveCik(TestSecServiceBase):
    @patch("spinoff_research.sec_service._paced_get")
    def test_resolves_known_ticker(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_TICKER_MAP_RAW)
        from spinoff_research.sec_service import resolve_cik
        result = resolve_cik("GEV")
        self.assertEqual(result["cik"], "0001996810")
        self.assertEqual(result["name"], "GE Vernova Inc.")

    @patch("spinoff_research.sec_service._paced_get")
    def test_returns_none_for_unknown_ticker(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_TICKER_MAP_RAW)
        from spinoff_research.sec_service import resolve_cik
        self.assertIsNone(resolve_cik("ZZZZZ_NOT_REAL"))

    @patch("spinoff_research.sec_service._paced_get")
    def test_ticker_lookup_is_case_insensitive(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_TICKER_MAP_RAW)
        from spinoff_research.sec_service import resolve_cik
        self.assertEqual(resolve_cik("gev")["cik"], "0001996810")

    @patch("spinoff_research.sec_service._paced_get")
    def test_second_call_uses_disk_cache_not_network(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_TICKER_MAP_RAW)
        from spinoff_research.sec_service import resolve_cik
        resolve_cik("GEV")
        resolve_cik("GE")
        # only the first lookup should have hit the network; second reads the cache file
        self.assertEqual(mock_get.call_count, 1)

    def test_missing_user_agent_raises(self):
        del os.environ["SEC_EDGAR_USER_AGENT"]
        from spinoff_research.sec_service import _user_agent, SecServiceError
        with self.assertRaises(SecServiceError):
            _user_agent()


class TestDiscoverFilings(TestSecServiceBase):
    @patch("spinoff_research.sec_service._paced_get")
    def test_filters_by_form_type(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_SUBMISSIONS_DOMESTIC)
        from spinoff_research.sec_service import discover_filings
        results = discover_filings("0001996810", form_types=["10-12B", "10-12B/A"])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(f.form_type in ("10-12B", "10-12B/A") for f in results))

    @patch("spinoff_research.sec_service._paced_get")
    def test_filters_by_date_window(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_SUBMISSIONS_DOMESTIC)
        from spinoff_research.sec_service import discover_filings
        results = discover_filings(
            "0001996810", form_types=["4"], filed_after="2025-01-01"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].filing_date, "2025-06-01")

    @patch("spinoff_research.sec_service._paced_get")
    def test_excludes_form_types_not_in_relevant_set_by_default(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_SUBMISSIONS_DOMESTIC)
        from spinoff_research.sec_service import discover_filings
        results = discover_filings("0001996810")  # no form_types override
        form_types_found = {f.form_type for f in results}
        self.assertNotIn("SC 13G", form_types_found)  # deliberately excluded — not in RELEVANT_FORM_TYPES

    @patch("spinoff_research.sec_service._paced_get")
    def test_builds_correct_sec_url(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_SUBMISSIONS_DOMESTIC)
        from spinoff_research.sec_service import discover_filings
        results = discover_filings("0001996810", form_types=["10-12B"])
        self.assertEqual(
            results[0].sec_url,
            "https://www.sec.gov/Archives/edgar/data/1996810/000119312524037526/d542465d1012b.htm",
        )

    @patch("spinoff_research.sec_service._paced_get")
    def test_maps_source_type_correctly(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_SUBMISSIONS_DOMESTIC)
        from spinoff_research.sec_service import discover_filings
        from spinoff_research.field_data_dictionary import SourceType
        results = discover_filings("0001996810", form_types=["10-12B"])
        self.assertEqual(results[0].source_type, SourceType.FORM_10)

    @patch("spinoff_research.sec_service._paced_get")
    def test_raw_document_url_strips_xsl_viewer_path_for_form4(self, mock_get):
        """
        Regression: confirmed live across all 3 pilots that the submissions
        API's primaryDocument for Form 4 filings is reported as
        'xslF345XNN/<file>.xml' — that path serves an SEC XSL-rendered HTML
        VIEW of the filing, not the raw machine-parseable XML, even though
        it ends in .xml. raw_document_url must strip the xsl subdirectory
        to reach the real file at the accession root.
        """
        mock_get.return_value = _mock_response(SAMPLE_SUBMISSIONS_DOMESTIC)
        from spinoff_research.sec_service import discover_filings
        results = discover_filings("0001996810", form_types=["4"])
        tx = results[0]
        self.assertIn("xslF345X02", tx.sec_url)  # sec_url still reflects the SEC-reported path as-is
        self.assertNotIn("xslF345X02", tx.raw_document_url)
        self.assertTrue(tx.raw_document_url.endswith(tx.primary_document.rsplit("/", 1)[-1]))

    @patch("spinoff_research.sec_service._paced_get")
    def test_raw_document_url_unchanged_when_no_xsl_path(self, mock_get):
        mock_get.return_value = _mock_response(SAMPLE_SUBMISSIONS_DOMESTIC)
        from spinoff_research.sec_service import discover_filings
        results = discover_filings("0001996810", form_types=["10-12B"])
        tx = results[0]
        self.assertEqual(tx.raw_document_url, tx.sec_url)

    @patch("spinoff_research.sec_service._paced_get")
    def test_foreign_private_issuer_20f_maps_to_10k_source_type(self, mock_get):
        """
        Regression test: Sanofi (INBX pilot's parent) files 20-F/6-K, not
        10-K/8-K, because it's a foreign private issuer. Discovered live
        during Phase 3 — the original form-type map returned zero parent
        filings for this pilot until 20-F/6-K were added.
        """
        mock_get.return_value = _mock_response(SAMPLE_SUBMISSIONS_FPI)
        from spinoff_research.sec_service import discover_filings
        from spinoff_research.field_data_dictionary import SourceType
        results = discover_filings("0001121404", form_types=["20-F", "6-K"])
        self.assertEqual(len(results), 3)
        annual_report = next(f for f in results if f.form_type == "20-F")
        self.assertEqual(annual_report.source_type, SourceType.FORM_10K)
        current_reports = [f for f in results if f.form_type == "6-K"]
        self.assertTrue(all(f.source_type == SourceType.FORM_8K for f in current_reports))


class TestDiscoverFilingsForTransaction(TestSecServiceBase):
    @patch("spinoff_research.sec_service._paced_get")
    def test_covers_domestic_parent_and_spinco(self, mock_get):
        def side_effect(url, timeout=20):
            if "1996810" in url:  # spinco (GEV)
                return _mock_response(SAMPLE_SUBMISSIONS_DOMESTIC)
            return _mock_response(SAMPLE_SUBMISSIONS_DOMESTIC)  # reuse fixture for parent too

        mock_get.side_effect = side_effect
        from spinoff_research.sec_service import discover_filings_for_transaction
        result = discover_filings_for_transaction(
            parent_cik="0000040545", spinoff_cik="0001996810",
            parent_name="GE", spinoff_name="GEV",
            announcement_date="2021-11-09", distribution_date="2024-04-02",
        )
        self.assertIn("spinco_form_10", result)
        self.assertIn("parent_periodic", result)
        self.assertGreaterEqual(len(result["spinco_form_10"]), 1)

    @patch("spinoff_research.sec_service._paced_get")
    def test_fpi_parent_produces_nonzero_parent_buckets(self, mock_get):
        """
        End-to-end regression for the FPI gap: before the 20-F/6-K fix, a
        transaction with a foreign-private-issuer parent (Sanofi/Inhibrx)
        silently returned 0 filings in both parent buckets.
        """
        def side_effect(url, timeout=20):
            if "1121404" in url:  # Sanofi (parent)
                return _mock_response(SAMPLE_SUBMISSIONS_FPI)
            return _mock_response(SAMPLE_SUBMISSIONS_DOMESTIC)  # spinco (INBX) — reuse domestic shape

        mock_get.side_effect = side_effect
        from spinoff_research.sec_service import discover_filings_for_transaction
        result = discover_filings_for_transaction(
            parent_cik="0001121404", spinoff_cik="0002007919",
            parent_name="Sanofi", spinoff_name="Inhibrx",
            announcement_date="2024-01-23", distribution_date="2024-05-30",
        )
        self.assertGreater(len(result["parent_event_driven"]), 0)
        self.assertGreater(len(result["parent_periodic"]), 0)


if __name__ == "__main__":
    unittest.main()

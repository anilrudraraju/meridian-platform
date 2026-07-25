import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


def _mock_response(json_data):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = json_data
    return resp


def _submissions(forms_with_dates):
    """Build a minimal submissions-JSON shape from [(form, filing_date), ...]."""
    return {
        "name": "Test Co",
        "filings": {
            "recent": {
                "form": [f for f, _ in forms_with_dates],
                "filingDate": [d for _, d in forms_with_dates],
                "reportDate": ["" for _ in forms_with_dates],
                "accessionNumber": [f"0001234567-24-{i:06d}" for i in range(len(forms_with_dates))],
                "primaryDocument": [f"doc{i}.htm" for i in range(len(forms_with_dates))],
            }
        },
    }


class TestBackwardExpansion(unittest.TestCase):
    def setUp(self):
        os.environ["SEC_EDGAR_USER_AGENT"] = "TestAgent test@example.com"
        self._tmpdir = tempfile.TemporaryDirectory()
        self._cache_patch = patch(
            "spinoff_research.sec_service._CACHE_DIR", Path(self._tmpdir.name)
        )
        self._cache_patch.start()

    def tearDown(self):
        self._cache_patch.stop()
        self._tmpdir.cleanup()

    @patch("spinoff_research.sec_service._paced_get")
    def test_no_expansion_needed_when_default_window_hits(self, mock_get):
        mock_get.return_value = _mock_response(
            _submissions([("10-K", "2024-01-15")])
        )
        from spinoff_research.sec_service import _discover_with_backward_expansion
        result = _discover_with_backward_expansion(
            "0001234567", "Test Co", form_types=["10-K"],
            filed_after="2024-01-01", filed_before="2024-06-01",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result.months_expanded, 0)
        self.assertFalse(result.exhausted)

    @patch("spinoff_research.sec_service._paced_get")
    def test_expands_backward_until_filing_found(self, mock_get):
        # Filing is 9 months before the default filed_after — mirrors the
        # real GE 10-K case (filed 9 months pre-announcement, well outside
        # the original fixed 15-month-lookback-only design point).
        mock_get.return_value = _mock_response(
            _submissions([("10-K", "2023-04-01")])
        )
        from spinoff_research.sec_service import _discover_with_backward_expansion
        result = _discover_with_backward_expansion(
            "0001234567", "Test Co", form_types=["10-K"],
            filed_after="2024-01-01", filed_before="2024-06-01",
        )
        self.assertEqual(len(result), 1)
        self.assertGreater(result.months_expanded, 0)
        self.assertFalse(result.exhausted)

    @patch("spinoff_research.sec_service._paced_get")
    def test_filed_before_never_moves_during_expansion(self, mock_get):
        """
        A filing that exists but AFTER filed_before must never be returned —
        expansion only walks filed_after backward, never filed_before
        forward. Otherwise a bucket meant to stop at "12 months
        post-distribution" (e.g. Form 4 aggregation) could silently start
        pulling filings from years later.
        """
        mock_get.return_value = _mock_response(
            _submissions([("10-K", "2024-09-01")])  # after filed_before
        )
        from spinoff_research.sec_service import _discover_with_backward_expansion
        result = _discover_with_backward_expansion(
            "0001234567", "Test Co", form_types=["10-K"],
            filed_after="2024-01-01", filed_before="2024-06-01",
        )
        self.assertEqual(len(result), 0)
        self.assertTrue(result.exhausted)  # ran out of backward room, found nothing in-window

    @patch("spinoff_research.sec_service._paced_get")
    def test_gives_up_after_24_months_and_flags_exhausted(self, mock_get):
        mock_get.return_value = _mock_response(_submissions([]))  # nothing ever found
        from spinoff_research.sec_service import (
            _discover_with_backward_expansion, _BACKWARD_EXPANSION_MAX_MONTHS,
        )
        result = _discover_with_backward_expansion(
            "0001234567", "Test Co", form_types=["10-K"],
            filed_after="2024-01-01", filed_before="2024-06-01",
        )
        self.assertEqual(len(result), 0)
        self.assertTrue(result.exhausted)
        self.assertEqual(result.months_expanded, _BACKWARD_EXPANSION_MAX_MONTHS)

    @patch("spinoff_research.sec_service._paced_get")
    def test_no_filed_after_means_no_expansion_possible(self, mock_get):
        """Form 10 buckets pass filed_after=None (already unbounded) — expansion must be a no-op, not an error."""
        mock_get.return_value = _mock_response(_submissions([]))
        from spinoff_research.sec_service import _discover_with_backward_expansion
        result = _discover_with_backward_expansion(
            "0001234567", "Test Co", form_types=["10-12B"],
            filed_after=None, filed_before=None,
        )
        self.assertEqual(len(result), 0)
        self.assertEqual(result.months_expanded, 0)
        self.assertFalse(result.exhausted)  # nothing to expand into — not the same as "gave up"

    @patch("spinoff_research.sec_service._paced_get")
    def test_bucket_result_is_list_like_for_backward_compatibility(self, mock_get):
        mock_get.return_value = _mock_response(
            _submissions([("10-K", "2024-01-15"), ("10-K/A", "2024-02-01")])
        )
        from spinoff_research.sec_service import _discover_with_backward_expansion
        result = _discover_with_backward_expansion(
            "0001234567", "Test Co", form_types=["10-K", "10-K/A"],
            filed_after="2024-01-01", filed_before="2024-06-01",
        )
        self.assertEqual(len(result), 2)
        self.assertTrue(result)
        for f in result:  # iterable
            self.assertIn(f.form_type, ("10-K", "10-K/A"))
        self.assertEqual(result[0].form_type, "10-K")  # indexable


if __name__ == "__main__":
    unittest.main()

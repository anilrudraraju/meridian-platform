import unittest
from unittest.mock import patch

from spinoff_research.tests.fixtures.sample_company_facts import (
    FACTS_SHARES_AGREE,
    FACTS_DEBT_CONFLICT,
    FACTS_REVENUE_DURATION_FALSE_CONFLICT,
    FACTS_SHARES_DRIFT_OVER_TIME,
    FACTS_EMPTY,
)
from spinoff_research.xbrl_service import (
    resolve_field,
    resolve_field_for_transaction,
    get_candidates,
    _parse_concept,
    XbrlServiceError,
)


class TestParseConcept(unittest.TestCase):
    def test_splits_taxonomy_and_tag(self):
        self.assertEqual(_parse_concept("us-gaap:LongTermDebt"), ("us-gaap", "LongTermDebt"))

    def test_bare_tag_defaults_to_us_gaap(self):
        self.assertEqual(_parse_concept("Revenues"), ("us-gaap", "Revenues"))


class TestGetCandidates(unittest.TestCase):
    def test_returns_empty_for_no_matching_concepts(self):
        result = get_candidates(FACTS_EMPTY, "spinoff_debt")
        self.assertEqual(result, [])

    def test_finds_candidates_across_multiple_concepts(self):
        result = get_candidates(FACTS_SHARES_AGREE, "spinoff_shares_outstanding")
        self.assertEqual(len(result), 2)
        concepts = {c.concept for c in result}
        self.assertIn("dei:EntityCommonStockSharesOutstanding", concepts)
        self.assertIn("us-gaap:CommonStockSharesOutstanding", concepts)

    def test_unknown_field_key_raises(self):
        with self.assertRaises(XbrlServiceError):
            get_candidates(FACTS_EMPTY, "not_a_real_field")

    def test_standard_taxonomy_not_flagged_as_extension(self):
        result = get_candidates(FACTS_SHARES_AGREE, "spinoff_shares_outstanding")
        self.assertTrue(all(not c.is_extension_tag for c in result))


class TestResolveFieldSelection(unittest.TestCase):
    def test_single_candidate_is_high_confidence(self):
        result = resolve_field(FACTS_DEBT_CONFLICT, "spinoff_debt", period_end="2024-06-30", forms=None)
        # both LongTermDebt and LongTermDebtNoncurrent match — not single;
        # use a facts dict with only one concept present instead
        single = {"facts": {"us-gaap": {"LongTermDebt": FACTS_DEBT_CONFLICT["facts"]["us-gaap"]["LongTermDebt"]}}}
        result = resolve_field(single, "spinoff_debt")
        self.assertEqual(result.status, "extracted_high_confidence")
        self.assertEqual(result.selected.value, 500000000)

    def test_agreeing_candidates_are_high_confidence(self):
        result = resolve_field(FACTS_SHARES_AGREE, "spinoff_shares_outstanding", period_end="2024-04-02")
        self.assertEqual(result.status, "extracted_high_confidence")
        self.assertEqual(result.selected.value, 274085523)
        self.assertEqual(len(result.rejected_candidates), 1)

    def test_disagreeing_candidates_same_period_are_conflicting(self):
        """Regression: LongTermDebt=$500M vs LongTermDebtNoncurrent=$480M, same period_end — a real conflict."""
        result = resolve_field(FACTS_DEBT_CONFLICT, "spinoff_debt", period_end="2024-06-30")
        self.assertEqual(result.status, "conflicting_sources")
        self.assertIsNone(result.selected)
        self.assertEqual(len(result.rejected_candidates), 2)
        self.assertIsNotNone(result.conflict_note)

    def test_same_period_end_different_period_start_is_not_a_false_conflict(self):
        """
        Regression for the bug found live against Grail: quarterly ($31.97M,
        period_start=2024-04-01) and YTD ($58.69M, period_start=2024-01-01)
        figures share period_end=2024-06-30 but are different periods, not
        competing values for the same period.
        """
        result = resolve_field(
            FACTS_REVENUE_DURATION_FALSE_CONFLICT, "last_year_sales", period_end="2024-06-30",
        )
        # must NOT be conflicting_sources — these are two different (start,end) groups
        self.assertNotEqual(result.status, "conflicting_sources")

    def test_no_candidates_is_not_found(self):
        result = resolve_field(FACTS_EMPTY, "spinoff_debt")
        self.assertEqual(result.status, "not_found")
        self.assertIsNone(result.selected)


class TestResolveFieldForTransactionSnapshot(unittest.TestCase):
    def test_anchors_to_distribution_date_not_most_recent(self):
        """
        Regression for the core bug: without anchoring, the most recent XBRL
        value (266.3M, 2026) would be selected instead of the value at
        distribution (274.1M, 2024-04-02).
        """
        result = resolve_field_for_transaction(
            FACTS_SHARES_DRIFT_OVER_TIME, "spinoff_shares_outstanding",
            distribution_date="2024-04-02",
        )
        self.assertEqual(result.selected.value, 274085523)
        self.assertNotEqual(result.selected.value, 266333581)

    def test_stale_only_candidates_return_not_found(self):
        """If nothing falls within tolerance of distribution_date, don't substitute a stale value."""
        result = resolve_field_for_transaction(
            FACTS_SHARES_DRIFT_OVER_TIME, "spinoff_shares_outstanding",
            distribution_date="2020-01-01",  # far before any candidate
        )
        self.assertEqual(result.status, "not_found")
        self.assertIsNone(result.selected)

    def test_snapshot_field_without_distribution_date_raises(self):
        with self.assertRaises(XbrlServiceError):
            resolve_field_for_transaction(FACTS_SHARES_AGREE, "spinoff_shares_outstanding")

    def test_no_candidates_at_all_is_not_found(self):
        result = resolve_field_for_transaction(FACTS_EMPTY, "spinoff_debt", distribution_date="2024-01-01")
        self.assertEqual(result.status, "not_found")


class TestResolveFieldForTransactionAnnual(unittest.TestCase):
    def test_selects_annual_duration_not_quarterly(self):
        """
        Regression: last_year_sales must select the ~365-day annual figure
        ($120M, FY2023), not the quarterly ($31.97M) or YTD ($58.69M)
        figures that happen to be more recent by filing date.
        """
        result = resolve_field_for_transaction(
            FACTS_REVENUE_DURATION_FALSE_CONFLICT, "last_year_sales",
            announcement_date="2024-07-01",
        )
        self.assertEqual(result.selected.value, 120000000)
        self.assertEqual(result.selected.period_start, "2023-01-01")
        self.assertEqual(result.selected.period_end, "2023-12-31")

    def test_annual_field_without_announcement_date_raises(self):
        with self.assertRaises(XbrlServiceError):
            resolve_field_for_transaction(FACTS_REVENUE_DURATION_FALSE_CONFLICT, "last_year_sales")

    def test_excludes_annual_periods_ending_after_announcement(self):
        """An annual period ending AFTER the announcement isn't 'last year' — it's the current or future year."""
        facts = {
            "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
                {"start": "2024-01-01", "end": "2024-12-31", "val": 999, "accn": "x", "form": "10-K", "filed": "2025-02-01"},
            ]}}}}
        }
        result = resolve_field_for_transaction(facts, "last_year_sales", announcement_date="2024-06-01")
        self.assertEqual(result.status, "not_found")

    def test_no_annual_duration_candidate_available_is_not_found(self):
        facts = {
            "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
                {"start": "2024-04-01", "end": "2024-06-30", "val": 100, "accn": "x", "form": "10-Q", "filed": "2024-08-01"},
            ]}}}}
        }
        result = resolve_field_for_transaction(facts, "last_year_sales", announcement_date="2024-07-01")
        self.assertEqual(result.status, "not_found")
        self.assertIn("annual-length", result.selection_reason)


if __name__ == "__main__":
    unittest.main()

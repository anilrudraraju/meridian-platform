import unittest

from spinoff_research.form4_parser import (
    parse_form4_xml,
    summarize_insider_buying,
    Form4ParseError,
    PURCHASE_CODE,
)
from spinoff_research.tests.fixtures.sample_form4 import (
    FORM4_SINGLE_SALE,
    FORM4_MULTI_PURCHASE,
    FORM4_OPTION_EXERCISE,
    FORM4_EMPTY_NONDERIVATIVE,
)


class TestParseForm4Xml(unittest.TestCase):
    def test_parses_single_sale_transaction(self):
        txs = parse_form4_xml(FORM4_SINGLE_SALE)
        self.assertEqual(len(txs), 1)
        self.assertEqual(txs[0].transaction_code, "S")
        self.assertEqual(txs[0].reporting_owner_name, "Abate Victor")

    def test_sale_is_not_a_purchase(self):
        txs = parse_form4_xml(FORM4_SINGLE_SALE)
        self.assertFalse(txs[0].is_open_market_purchase)

    def test_parses_multiple_transactions_in_one_filing(self):
        """Regression: confirmed live that one filing can report several
        transactions for the same owner (split across custodial sub-accounts)."""
        txs = parse_form4_xml(FORM4_MULTI_PURCHASE)
        self.assertEqual(len(txs), 2)

    def test_multi_purchase_transactions_are_purchases(self):
        txs = parse_form4_xml(FORM4_MULTI_PURCHASE)
        self.assertTrue(all(tx.is_open_market_purchase for tx in txs))

    def test_sub_account_distinction_preserved(self):
        txs = parse_form4_xml(FORM4_MULTI_PURCHASE)
        notes = {tx.indirect_ownership_note for tx in txs}
        self.assertEqual(notes, {"By Child A", "By Child B"})

    def test_option_exercise_is_not_a_purchase(self):
        """Regression: code 'M' (option exercise) increases shares owned
        but must never count as insider buying — only 'P' does."""
        txs = parse_form4_xml(FORM4_OPTION_EXERCISE)
        self.assertEqual(txs[0].transaction_code, "M")
        self.assertFalse(txs[0].is_open_market_purchase)

    def test_empty_nonderivative_table_returns_no_transactions(self):
        """Regression: confirmed live a filing can be derivative-only."""
        txs = parse_form4_xml(FORM4_EMPTY_NONDERIVATIVE)
        self.assertEqual(txs, [])

    def test_malformed_xml_raises_form4_parse_error(self):
        with self.assertRaises(Form4ParseError):
            parse_form4_xml(b"<not valid xml")

    def test_issuer_and_owner_fields_extracted(self):
        txs = parse_form4_xml(FORM4_SINGLE_SALE)
        tx = txs[0]
        self.assertEqual(tx.issuer_cik, "0001996810")
        self.assertEqual(tx.issuer_ticker, "GEV")
        self.assertEqual(tx.reporting_owner_cik, "0002005215")
        self.assertTrue(tx.is_officer)
        self.assertFalse(tx.is_director)
        self.assertEqual(tx.officer_title, "Chief Executive Officer, Wind")

    def test_shares_and_price_parsed_as_floats(self):
        txs = parse_form4_xml(FORM4_SINGLE_SALE)
        self.assertEqual(txs[0].shares, 4819.0)
        self.assertEqual(txs[0].price_per_share, 948.08)

    def test_document_id_carried_through(self):
        txs = parse_form4_xml(FORM4_SINGLE_SALE, document_id="0001234567-24-000001")
        self.assertEqual(txs[0].document_id, "0001234567-24-000001")


class TestSummarizeInsiderBuying(unittest.TestCase):
    def test_purchase_within_window_counted(self):
        txs = parse_form4_xml(FORM4_MULTI_PURCHASE)
        summary = summarize_insider_buying(txs, "2024-11-01", "2024-11-30")
        self.assertTrue(summary.any_buying)
        self.assertEqual(summary.distinct_buyer_count, 1)  # same owner, 2 transactions -> 1 buyer

    def test_purchase_outside_window_excluded(self):
        txs = parse_form4_xml(FORM4_MULTI_PURCHASE)
        summary = summarize_insider_buying(txs, "2025-01-01", "2025-03-01")
        self.assertFalse(summary.any_buying)
        self.assertEqual(summary.distinct_buyer_count, 0)

    def test_sale_never_counted_as_buying(self):
        txs = parse_form4_xml(FORM4_SINGLE_SALE)
        summary = summarize_insider_buying(txs, "2020-01-01", "2030-01-01")
        self.assertFalse(summary.any_buying)

    def test_multiple_transactions_same_owner_count_as_one_buyer(self):
        """
        Regression: distinct_buyer_count must dedupe by reporting_owner_cik,
        not count transactions — confirmed live one INBX filing had 4
        purchase transactions for a single owner across sub-accounts.
        """
        txs = parse_form4_xml(FORM4_MULTI_PURCHASE)
        summary = summarize_insider_buying(txs, "2024-01-01", "2024-12-31")
        self.assertEqual(summary.distinct_buyer_count, 1)
        self.assertEqual(len(summary.purchase_transactions), 2)

    def test_cluster_threshold_configurable(self):
        txs = parse_form4_xml(FORM4_MULTI_PURCHASE)
        summary = summarize_insider_buying(txs, "2024-01-01", "2024-12-31", cluster_threshold=1)
        self.assertTrue(summary.is_cluster)

    def test_below_cluster_threshold_not_a_cluster(self):
        txs = parse_form4_xml(FORM4_MULTI_PURCHASE)
        summary = summarize_insider_buying(txs, "2024-01-01", "2024-12-31", cluster_threshold=3)
        self.assertFalse(summary.is_cluster)

    def test_aggregates_across_multiple_filings(self):
        """Real usage: transactions from many separate Form 4 filings get
        pooled before summarizing, not summarized one filing at a time."""
        filing_a = parse_form4_xml(FORM4_MULTI_PURCHASE)
        filing_b = parse_form4_xml(FORM4_SINGLE_SALE)
        filing_c = parse_form4_xml(FORM4_OPTION_EXERCISE)
        pooled = filing_a + filing_b + filing_c
        summary = summarize_insider_buying(pooled, "2024-01-01", "2024-12-31")
        self.assertEqual(summary.distinct_buyer_count, 1)  # only the purchase filing counts


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from spinoff_research.db import init_db
from spinoff_research.models import Company, SpinoffTransaction
from spinoff_research.repository import (
    get_or_create_company,
    get_or_create_transaction,
    get_transaction,
    list_transactions,
)


class TestRepository(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.conn = init_db(Path(self._tmpdir.name) / "test.db")

    def tearDown(self):
        self.conn.close()
        self._tmpdir.cleanup()

    def test_creates_new_company(self):
        c = get_or_create_company(self.conn, Company(name="GE Vernova Inc.", ticker="GEV", cik="0001996810"))
        self.assertIsNotNone(c.company_id)

    def test_reuses_existing_company_on_second_call(self):
        c1 = get_or_create_company(self.conn, Company(name="GE Vernova Inc.", ticker="GEV", cik="0001996810"))
        c2 = get_or_create_company(self.conn, Company(name="GE Vernova Inc.", ticker="GEV", cik="0001996810"))
        self.assertEqual(c1.company_id, c2.company_id)
        count = self.conn.execute("SELECT COUNT(*) AS c FROM companies").fetchone()["c"]
        self.assertEqual(count, 1)

    def test_fills_in_missing_fields_without_overwriting_existing(self):
        get_or_create_company(self.conn, Company(name="GE Vernova Inc.", ticker="GEV"))
        updated = get_or_create_company(
            self.conn, Company(name="GE Vernova Inc.", ticker="GEV", cik="0001996810", sector="Industrials")
        )
        self.assertEqual(updated.cik, "0001996810")
        self.assertEqual(updated.sector, "Industrials")

        # a later call with a DIFFERENT cik must not clobber the stored one
        again = get_or_create_company(
            self.conn, Company(name="GE Vernova Inc.", ticker="GEV", cik="9999999999")
        )
        self.assertEqual(again.cik, "0001996810")

    def test_creates_transaction_and_both_companies(self):
        tx = SpinoffTransaction(
            parent=Company(name="General Electric", ticker="GE"),
            spinoff=Company(name="GE Vernova Inc.", ticker="GEV"),
            spinoff_date="2024-04-02",
        )
        saved = get_or_create_transaction(self.conn, tx)
        self.assertIsNotNone(saved.transaction_id)
        self.assertIsNotNone(saved.parent.company_id)
        self.assertIsNotNone(saved.spinoff.company_id)
        self.assertNotEqual(saved.parent.company_id, saved.spinoff.company_id)

    def test_transaction_idempotent_on_same_company_pair(self):
        tx = SpinoffTransaction(
            parent=Company(name="General Electric", ticker="GE"),
            spinoff=Company(name="GE Vernova Inc.", ticker="GEV"),
        )
        first = get_or_create_transaction(self.conn, tx)
        second = get_or_create_transaction(self.conn, tx)
        self.assertEqual(first.transaction_id, second.transaction_id)
        count = self.conn.execute("SELECT COUNT(*) AS c FROM spinoff_transactions").fetchone()["c"]
        self.assertEqual(count, 1)

    def test_get_transaction_loads_full_company_details(self):
        tx = SpinoffTransaction(
            parent=Company(name="General Electric", ticker="GE", cik="0000040545"),
            spinoff=Company(name="GE Vernova Inc.", ticker="GEV", cik="0001996810"),
            spinoff_date="2024-04-02",
        )
        saved = get_or_create_transaction(self.conn, tx)
        loaded = get_transaction(self.conn, saved.transaction_id)
        self.assertEqual(loaded.parent.cik, "0000040545")
        self.assertEqual(loaded.spinoff.ticker, "GEV")
        self.assertEqual(loaded.spinoff_date, "2024-04-02")

    def test_get_transaction_returns_none_for_unknown_id(self):
        self.assertIsNone(get_transaction(self.conn, 99999))

    def test_spinoff_date_backfilled_when_missing_on_reinsert(self):
        tx_no_date = SpinoffTransaction(
            parent=Company(name="General Electric", ticker="GE"),
            spinoff=Company(name="GE Vernova Inc.", ticker="GEV"),
        )
        first = get_or_create_transaction(self.conn, tx_no_date)
        self.assertIsNone(first.spinoff_date)

        tx_with_date = SpinoffTransaction(
            parent=Company(name="General Electric", ticker="GE"),
            spinoff=Company(name="GE Vernova Inc.", ticker="GEV"),
            spinoff_date="2024-04-02",
        )
        second = get_or_create_transaction(self.conn, tx_with_date)
        self.assertEqual(second.transaction_id, first.transaction_id)
        loaded = get_transaction(self.conn, first.transaction_id)
        self.assertEqual(loaded.spinoff_date, "2024-04-02")

    def test_announcement_date_persisted_and_backfilled(self):
        tx_no_date = SpinoffTransaction(
            parent=Company(name="General Electric", ticker="GE"),
            spinoff=Company(name="GE Vernova Inc.", ticker="GEV"),
        )
        first = get_or_create_transaction(self.conn, tx_no_date)
        self.assertIsNone(first.announcement_date)

        tx_with_date = SpinoffTransaction(
            parent=Company(name="General Electric", ticker="GE"),
            spinoff=Company(name="GE Vernova Inc.", ticker="GEV"),
            announcement_date="2021-11-09",
        )
        second = get_or_create_transaction(self.conn, tx_with_date)
        self.assertEqual(second.transaction_id, first.transaction_id)
        loaded = get_transaction(self.conn, first.transaction_id)
        self.assertEqual(loaded.announcement_date, "2021-11-09")

    def test_list_transactions_empty(self):
        self.assertEqual(list_transactions(self.conn), [])

    def test_list_transactions_returns_all(self):
        get_or_create_transaction(self.conn, SpinoffTransaction(
            parent=Company(name="Parent A", ticker="PA"), spinoff=Company(name="Spinco A", ticker="SA"),
        ))
        get_or_create_transaction(self.conn, SpinoffTransaction(
            parent=Company(name="Parent B", ticker="PB"), spinoff=Company(name="Spinco B", ticker="SB"),
        ))
        txs = list_transactions(self.conn)
        self.assertEqual(len(txs), 2)
        self.assertEqual({t.spinoff.ticker for t in txs}, {"SA", "SB"})


if __name__ == "__main__":
    unittest.main()

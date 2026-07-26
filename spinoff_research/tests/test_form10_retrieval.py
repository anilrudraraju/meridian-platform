import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from spinoff_research.db import init_db
from spinoff_research.form10_retrieval import (
    Form10NotFoundError,
    find_relevant_sections,
    get_or_ingest_form10,
)
from spinoff_research.models import Company, SpinoffTransaction
from spinoff_research.repository import get_or_create_transaction
from spinoff_research.sec_service import DiscoveredFiling, SecServiceError


def _make_filing(**overrides):
    defaults = dict(
        cik="0000000002", company_name="Spinco Inc", form_type="10-12B/A",
        filing_date="2024-03-01", accession_number="000119312524059354",
        accession_number_dashed="0001193125-24-059354", primary_document="cover1012ba.htm",
    )
    defaults.update(overrides)
    return DiscoveredFiling(**defaults)


class TestGetOrIngestForm10(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.conn = init_db(Path(self._tmpdir.name) / "test.db")
        tx = SpinoffTransaction(
            parent=Company(name="Parent Co", ticker="P", cik="0000000001"),
            spinoff=Company(name="Spinco Inc", ticker="S", cik="0000000002"),
        )
        self.txn = get_or_create_transaction(self.conn, tx)

    def tearDown(self):
        self.conn.close()
        self._tmpdir.cleanup()

    def test_returns_existing_document_id_without_re_ingesting(self):
        self.conn.execute(
            "INSERT INTO documents (transaction_id, accession_number, primary_document_name, filing_type, filing_date, source_type) "
            "VALUES (?, 'acc-1', 'cover.htm', '10-12B', '2024-02-01', 'FORM_10')",
            (self.txn.transaction_id,),
        )
        self.conn.commit()
        doc_ids = get_or_ingest_form10(self.conn, self.txn.transaction_id, "0000000002")
        self.assertEqual(len(doc_ids), 1)
        self.assertIsNotNone(doc_ids[0])

    def test_returns_both_primary_and_exhibit_when_both_already_ingested(self):
        self.conn.execute(
            "INSERT INTO documents (transaction_id, accession_number, primary_document_name, filing_type, filing_date, source_type) "
            "VALUES (?, 'acc-1', 'cover.htm', '10-12B', '2024-02-01', 'FORM_10')",
            (self.txn.transaction_id,),
        )
        self.conn.execute(
            "INSERT INTO documents (transaction_id, accession_number, primary_document_name, filing_type, filing_date, source_type) "
            "VALUES (?, 'acc-1', 'ex991.htm', '10-12B', '2024-02-01', 'FORM_10')",
            (self.txn.transaction_id,),
        )
        self.conn.commit()
        doc_ids = get_or_ingest_form10(self.conn, self.txn.transaction_id, "0000000002")
        self.assertEqual(len(doc_ids), 2)

    @patch("spinoff_research.form10_retrieval.discover_filings")
    def test_raises_when_no_form10_exists(self, mock_discover):
        mock_discover.return_value = []
        with self.assertRaises(Form10NotFoundError):
            get_or_ingest_form10(self.conn, self.txn.transaction_id, "0000000002")

    @patch("spinoff_research.form10_retrieval.discover_filings")
    def test_raises_on_edgar_query_failure(self, mock_discover):
        mock_discover.side_effect = SecServiceError("EDGAR unreachable")
        with self.assertRaises(Form10NotFoundError):
            get_or_ingest_form10(self.conn, self.txn.transaction_id, "0000000002")

    @patch("spinoff_research.form10_retrieval._find_exhibit_99_1_filename")
    @patch("spinoff_research.form10_retrieval.ingest_filing")
    @patch("spinoff_research.form10_retrieval.discover_filings")
    def test_picks_most_recently_filed_when_multiple_amendments_exist(self, mock_discover, mock_ingest, mock_find_exhibit):
        older = _make_filing(filing_date="2024-01-01")
        newer = _make_filing(filing_date="2024-03-01")
        mock_discover.return_value = [older, newer]
        mock_ingest.return_value = MagicMock(document_id=99)
        mock_find_exhibit.return_value = None

        doc_ids = get_or_ingest_form10(self.conn, self.txn.transaction_id, "0000000002")
        self.assertEqual(doc_ids, [99])
        mock_ingest.assert_called_once_with(self.conn, self.txn.transaction_id, None, newer)

    @patch("spinoff_research.form10_retrieval._find_exhibit_99_1_filename")
    @patch("spinoff_research.form10_retrieval.ingest_filing")
    @patch("spinoff_research.form10_retrieval.discover_filings")
    def test_no_exhibit_found_returns_primary_document_only(self, mock_discover, mock_ingest, mock_find_exhibit):
        """Some filers put Item 5 content directly in the primary document —
        this must not be treated as an error."""
        mock_discover.return_value = [_make_filing()]
        mock_ingest.return_value = MagicMock(document_id=1)
        mock_find_exhibit.return_value = None

        doc_ids = get_or_ingest_form10(self.conn, self.txn.transaction_id, "0000000002")
        self.assertEqual(doc_ids, [1])
        mock_ingest.assert_called_once()

    @patch("spinoff_research.form10_retrieval._find_exhibit_99_1_filename")
    @patch("spinoff_research.form10_retrieval.ingest_filing")
    @patch("spinoff_research.form10_retrieval.discover_filings")
    def test_exhibit_ingested_with_its_own_filename_not_primarys(self, mock_discover, mock_ingest, mock_find_exhibit):
        """Regression for the real bug found in live pilot validation: the
        exhibit must be ingested as a DiscoveredFiling carrying the
        EXHIBIT's filename, not the primary document's — otherwise
        ingestion.py's exhibit_number/primary_document_name bookkeeping
        (and its UNIQUE constraint) silently attributes the exhibit's
        content to the cover page's identity."""
        primary_filing = _make_filing(primary_document="d542465d1012ba.htm")
        mock_discover.return_value = [primary_filing]
        mock_find_exhibit.return_value = "d542465dex991.htm"
        mock_ingest.side_effect = [MagicMock(document_id=1), MagicMock(document_id=2)]

        doc_ids = get_or_ingest_form10(self.conn, self.txn.transaction_id, "0000000002")
        self.assertEqual(doc_ids, [1, 2])

        self.assertEqual(mock_ingest.call_count, 2)
        exhibit_call_args = mock_ingest.call_args_list[1][0]
        exhibit_filing_arg = exhibit_call_args[3]
        self.assertEqual(exhibit_filing_arg.primary_document, "d542465dex991.htm")
        self.assertNotEqual(exhibit_filing_arg.primary_document, primary_filing.primary_document)

    @patch("spinoff_research.form10_retrieval._find_exhibit_99_1_filename")
    @patch("spinoff_research.form10_retrieval.ingest_filing")
    @patch("spinoff_research.form10_retrieval.discover_filings")
    def test_exhibit_ingestion_failure_does_not_fail_whole_call(self, mock_discover, mock_ingest, mock_find_exhibit):
        """Exhibit fetch is best-effort — a failure there shouldn't discard
        an already-successfully-ingested primary document."""
        from spinoff_research.ingestion import IngestionError

        mock_discover.return_value = [_make_filing()]
        mock_find_exhibit.return_value = "some_exhibit.htm"
        mock_ingest.side_effect = [MagicMock(document_id=1), IngestionError("404")]

        doc_ids = get_or_ingest_form10(self.conn, self.txn.transaction_id, "0000000002")
        self.assertEqual(doc_ids, [1])


class TestFindExhibit991Filename(unittest.TestCase):
    @patch("spinoff_research.form10_retrieval._paced_get")
    def test_picks_largest_when_multiple_ex99_variants_match(self, mock_get):
        """Regression for Inhibrx's real accession: an ex99-2 press release
        (9KB) coexists with the actual ex99-1 Information Statement in a
        THIRD naming convention (exh99x1, ~5MB). Size, not name pattern
        alone, must disambiguate."""
        from spinoff_research.form10_retrieval import _find_exhibit_99_1_filename

        mock_get.return_value.json.return_value = {
            "directory": {"item": [
                {"name": "tm243190-12_1012ba.htm", "size": "79010"},
                {"name": "tm243190d12_exh99x1.htm", "size": "5000370"},
                {"name": "tm243190d13_ex99-2.htm", "size": "9306"},
            ]}
        }
        filing = _make_filing()
        result = _find_exhibit_99_1_filename(filing)
        self.assertEqual(result, "tm243190d12_exh99x1.htm")

    @patch("spinoff_research.form10_retrieval._paced_get")
    def test_returns_none_when_no_exhibit_matches(self, mock_get):
        from spinoff_research.form10_retrieval import _find_exhibit_99_1_filename

        mock_get.return_value.json.return_value = {
            "directory": {"item": [{"name": "cover.htm", "size": "1000"}]}
        }
        result = _find_exhibit_99_1_filename(_make_filing())
        self.assertIsNone(result)

    @patch("spinoff_research.form10_retrieval._paced_get")
    def test_returns_none_on_index_fetch_failure(self, mock_get):
        from spinoff_research.form10_retrieval import _find_exhibit_99_1_filename

        mock_get.side_effect = SecServiceError("network error")
        result = _find_exhibit_99_1_filename(_make_filing())
        self.assertIsNone(result)


class TestFindRelevantSections(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.conn = init_db(Path(self._tmpdir.name) / "test.db")
        tx = SpinoffTransaction(parent=Company(name="Parent"), spinoff=Company(name="Spinco"))
        self.txn = get_or_create_transaction(self.conn, tx)
        self.conn.execute(
            "INSERT INTO documents (transaction_id, filing_type, source_type) VALUES (?, '10-12B', 'FORM_10')",
            (self.txn.transaction_id,),
        )
        self.doc_id = self.conn.execute("SELECT document_id FROM documents").fetchone()["document_id"]

    def tearDown(self):
        self.conn.close()
        self._tmpdir.cleanup()

    def _add_section(self, title, content, idx, doc_id=None):
        self.conn.execute(
            "INSERT INTO document_sections (document_id, section_title, chunk_index, content) VALUES (?, ?, ?, ?)",
            (doc_id or self.doc_id, title, idx, content),
        )
        self.conn.commit()

    def test_excludes_table_sections(self):
        self._add_section("table:Financials", "Chief Executive Officer compensation table", 0)
        self._add_section("prose:0", "Our Chief Executive Officer has served since 2020.", 1)
        results = find_relevant_sections(self.conn, [self.doc_id], ["chief executive officer"])
        self.assertEqual(len(results), 1)
        self.assertIn("served since 2020", results[0].content)

    def test_ranks_by_match_count_descending(self):
        self._add_section("prose:0", "biography text with chief executive officer mention", 0)
        self._add_section("prose:1", "biography text with chief executive officer and prior to the separation and served as details", 1)
        results = find_relevant_sections(
            self.conn, [self.doc_id],
            ["biography", "chief executive officer", "prior to the separation", "served as"],
        )
        self.assertEqual(len(results), 2)
        self.assertIn("prior to the separation", results[0].content)

    def test_respects_max_sections_cap(self):
        for i in range(10):
            self._add_section(f"prose:{i}", "management biography text", i)
        results = find_relevant_sections(self.conn, [self.doc_id], ["management"], max_sections=3)
        self.assertEqual(len(results), 3)

    def test_no_matches_returns_empty_list(self):
        self._add_section("prose:0", "unrelated capital structure discussion", 0)
        results = find_relevant_sections(self.conn, [self.doc_id], ["chief executive officer"])
        self.assertEqual(results, [])

    def test_searches_across_multiple_document_ids(self):
        """The real-world case: Item 5 biography content lives in the
        Exhibit 99.1 document, not the primary Form 10 — both document_ids
        must be searched together."""
        cur = self.conn.execute(
            "INSERT INTO documents (transaction_id, filing_type, source_type) VALUES (?, '10-12B', 'FORM_10')",
            (self.txn.transaction_id,),
        )
        self.conn.commit()
        exhibit_doc_id = cur.lastrowid

        self._add_section("prose:0", "cover page boilerplate, no biography content", 0, doc_id=self.doc_id)
        self._add_section("prose:0", "Jane Doe has served as Chief Executive Officer since 2020.", 0, doc_id=exhibit_doc_id)

        results = find_relevant_sections(self.conn, [self.doc_id, exhibit_doc_id], ["chief executive officer"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document_id, exhibit_doc_id)


if __name__ == "__main__":
    unittest.main()

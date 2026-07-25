import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from spinoff_research.db import init_db
from spinoff_research.ingestion import ingest_filing, IngestionError
from spinoff_research.sec_service import DiscoveredFiling

_SAMPLE_HTML = b"""<html><body>
<b>CAPITALIZATION</b>
<table>
  <tr><td></td><td>Historical</td><td>Pro Forma</td></tr>
  <tr><td>Total debt</td><td>129</td><td>129</td></tr>
  <tr><td>Total equity</td><td>7,416</td><td>8,004</td></tr>
</table>
<p>This filing describes the spin-off transaction in narrative form, covering risk factors and business strategy at some length so the prose chunker has real content to split.</p>
</body></html>"""

_SAMPLE_FORM4_XML = b"""<?xml version="1.0"?>
<ownershipDocument>
    <issuer><issuerCik>0001996810</issuerCik><issuerName>GE Vernova Inc.</issuerName><issuerTradingSymbol>GEV</issuerTradingSymbol></issuer>
    <reportingOwner><reportingOwnerId><rptOwnerCik>0002005215</rptOwnerCik><rptOwnerName>Abate Victor</rptOwnerName></reportingOwnerId></reportingOwner>
    <nonDerivativeTable></nonDerivativeTable>
</ownershipDocument>"""


def _mock_response(content: bytes):
    resp = MagicMock()
    resp.status_code = 200
    resp.content = content
    return resp


def _sample_filing(primary_document="d542465dex991.htm", form_type="10-12B EX-99.1"):
    return DiscoveredFiling(
        cik="0001996810", company_name="GE Vernova Inc.", form_type=form_type,
        filing_date="2024-02-15", accession_number="000119312524037526",
        accession_number_dashed="0001193125-24-037526",
        primary_document=primary_document,
    )


class TestIngestFiling(unittest.TestCase):
    def setUp(self):
        os.environ["SEC_EDGAR_USER_AGENT"] = "TestAgent test@example.com"
        self._tmpdir = tempfile.TemporaryDirectory()
        self.conn = init_db(Path(self._tmpdir.name) / "test.db")
        self._cache_patch = patch(
            "spinoff_research.ingestion._CACHE_DIR", Path(self._tmpdir.name) / "cache"
        )
        self._cache_patch.start()
        self.conn.execute("INSERT INTO companies (company_id, name, ticker) VALUES (1, 'Parent', 'P')")
        self.conn.execute("INSERT INTO companies (company_id, name, ticker) VALUES (2, 'Spinco', 'S')")
        self.conn.execute(
            "INSERT INTO spinoff_transactions (transaction_id, parent_company_id, spinoff_company_id) "
            "VALUES (1, 1, 2)"
        )
        self.conn.commit()

    def tearDown(self):
        self._cache_patch.stop()
        self.conn.close()
        self._tmpdir.cleanup()

    @patch("spinoff_research.ingestion._paced_get")
    def test_ingests_html_filing_and_writes_documents_row(self, mock_get):
        mock_get.return_value = _mock_response(_SAMPLE_HTML)
        result = ingest_filing(self.conn, transaction_id=1, company_id=2, filing=_sample_filing())
        self.assertIsNotNone(result.document_id)
        row = self.conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (result.document_id,)
        ).fetchone()
        self.assertEqual(row["transaction_id"], 1)
        self.assertEqual(row["ingestion_status"], "downloaded")
        self.assertEqual(row["text_extraction_status"], "extracted")

    @patch("spinoff_research.ingestion._paced_get")
    def test_caches_raw_bytes_to_disk(self, mock_get):
        mock_get.return_value = _mock_response(_SAMPLE_HTML)
        result = ingest_filing(self.conn, transaction_id=1, company_id=2, filing=_sample_filing())
        row = self.conn.execute(
            "SELECT local_path FROM documents WHERE document_id = ?", (result.document_id,)
        ).fetchone()
        self.assertTrue(Path(row["local_path"]).exists())
        self.assertEqual(Path(row["local_path"]).read_bytes(), _SAMPLE_HTML)

    @patch("spinoff_research.ingestion._paced_get")
    def test_second_ingest_of_same_content_hits_cache(self, mock_get):
        mock_get.return_value = _mock_response(_SAMPLE_HTML)
        first = ingest_filing(self.conn, transaction_id=1, company_id=2, filing=_sample_filing())
        second = ingest_filing(self.conn, transaction_id=1, company_id=2, filing=_sample_filing())
        self.assertFalse(first.was_cached)
        self.assertTrue(second.was_cached)
        self.assertEqual(first.document_id, second.document_id)
        # only ONE row in documents despite two ingest_filing calls
        count = self.conn.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"]
        self.assertEqual(count, 1)

    @patch("spinoff_research.ingestion._paced_get")
    def test_extracts_table_sections(self, mock_get):
        mock_get.return_value = _mock_response(_SAMPLE_HTML)
        result = ingest_filing(self.conn, transaction_id=1, company_id=2, filing=_sample_filing())
        table_sections = self.conn.execute(
            "SELECT * FROM document_sections WHERE document_id = ? AND section_title LIKE 'table:%'",
            (result.document_id,),
        ).fetchall()
        self.assertEqual(len(table_sections), 1)
        self.assertIn("CAPITALIZATION", table_sections[0]["section_title"])
        self.assertIn("Total debt", table_sections[0]["content"])
        self.assertIn("129", table_sections[0]["content"])

    @patch("spinoff_research.ingestion._paced_get")
    def test_extracts_prose_sections_with_char_offsets(self, mock_get):
        mock_get.return_value = _mock_response(_SAMPLE_HTML)
        result = ingest_filing(self.conn, transaction_id=1, company_id=2, filing=_sample_filing())
        prose_sections = self.conn.execute(
            "SELECT * FROM document_sections WHERE document_id = ? AND section_title LIKE 'prose:%'",
            (result.document_id,),
        ).fetchall()
        self.assertGreaterEqual(len(prose_sections), 1)
        self.assertIsNotNone(prose_sections[0]["char_start"])
        self.assertIsNotNone(prose_sections[0]["char_end"])
        self.assertIn("spin-off transaction", prose_sections[0]["content"])

    @patch("spinoff_research.ingestion._paced_get")
    def test_form4_xml_is_cached_but_not_sectioned(self, mock_get):
        """
        Regression: Form 4 XML must be stored (documents row + cached raw
        bytes) but NOT run through table/prose extraction — it's
        structured data with no prose/table distinction, and
        form4_parser.py is the correct tool to read it, not
        html_table_parser.py.
        """
        mock_get.return_value = _mock_response(_SAMPLE_FORM4_XML)
        filing = _sample_filing(primary_document="wk-form4.xml", form_type="4")
        result = ingest_filing(self.conn, transaction_id=1, company_id=2, filing=filing)
        self.assertEqual(result.section_count, 0)
        self.assertEqual(result.text_extraction_status, "not_applicable")
        row = self.conn.execute(
            "SELECT * FROM documents WHERE document_id = ?", (result.document_id,)
        ).fetchone()
        self.assertEqual(row["ingestion_status"], "downloaded")

    @patch("spinoff_research.ingestion._paced_get")
    def test_source_type_stored_from_filing(self, mock_get):
        mock_get.return_value = _mock_response(_SAMPLE_HTML)
        from spinoff_research.field_data_dictionary import SourceType
        filing = _sample_filing()
        filing.source_type = SourceType.FORM_10
        result = ingest_filing(self.conn, transaction_id=1, company_id=2, filing=filing)
        row = self.conn.execute(
            "SELECT source_type FROM documents WHERE document_id = ?", (result.document_id,)
        ).fetchone()
        self.assertEqual(row["source_type"], "FORM_10")

    @patch("spinoff_research.ingestion._paced_get")
    def test_download_failure_raises_ingestion_error(self, mock_get):
        from spinoff_research.sec_service import SecServiceError
        mock_get.side_effect = SecServiceError("network failure")
        with self.assertRaises(IngestionError):
            ingest_filing(self.conn, transaction_id=1, company_id=2, filing=_sample_filing())


if __name__ == "__main__":
    unittest.main()

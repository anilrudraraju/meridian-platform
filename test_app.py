#!/usr/bin/env python3
"""
Automated test suite for app.py helper functions.
Tests pure / side-effect-free logic only — no Streamlit UI, no API calls.

Run:
    python3 test_app.py
"""
import sys
import unittest
from unittest.mock import MagicMock
import hashlib

# ── 1. Mock streamlit BEFORE importing app ────────────────────────────────────

class _MockSessionState(dict):
    """Supports both dict-key and attribute access, like st.session_state."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

_st = MagicMock()
# active_layer="" ensures no layer UI block runs during import
_st.session_state = _MockSessionState(active_layer="")
# Neutralise @st.cache_resource so it returns the raw function unchanged
_st.cache_resource = lambda fn: fn
# st.secrets must be a real dict — os.environ rejects MagicMock values
_st.secrets = {"OPENAI_API_KEY": "test-key-not-used"}
# Buttons must return False — MagicMock is truthy and would "click" every button,
# causing sidebar handlers to overwrite active_layer during import
_st.button.return_value = False
# columns/tabs must be unpackable; return a tuple of MagicMocks sized to match usage
_st.columns.side_effect = lambda n, **kw: tuple(MagicMock() for _ in range(n if isinstance(n, int) else len(n)))
sys.modules["streamlit"] = _st

# Mock heavy optional deps that may not be present / we don't want to hit
for _mod in [
    "chromadb", "openai", "yfinance",
    "pypdf", "pdfplumber",
    "sentence_transformers",
    "rouge_score", "rouge_score.rouge_scorer",
    "sklearn", "sklearn.metrics", "sklearn.metrics.pairwise",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import app  # noqa: E402 — must come after mocks


# ══════════════════════════════════════════════════════════════════════════════
# get_chunking_config
# ══════════════════════════════════════════════════════════════════════════════

class TestGetChunkingConfig(unittest.TestCase):

    def test_10k_config_values(self):
        c = app.get_chunking_config("10-K")
        self.assertEqual(c["business_chunk_size"], 1500)
        self.assertEqual(c["business_overlap"], 300)
        self.assertTrue(c["xbrl_expected"])
        self.assertFalse(c["has_extra_sections"])
        self.assertFalse(c["check_predecessor"])
        self.assertTrue(c["skip_financial_stmts"])

    def test_10q_config_values(self):
        c = app.get_chunking_config("10-Q")
        self.assertEqual(c["business_chunk_size"], 1500)
        self.assertTrue(c["xbrl_expected"])
        self.assertFalse(c["has_extra_sections"])
        self.assertTrue(c["skip_financial_stmts"])

    def test_form_10_label_normalised_to_internal_key(self):
        """UI label 'Form 10' must produce the same config as internal key '10'."""
        self.assertEqual(app.get_chunking_config("Form 10"), app.get_chunking_config("10"))

    def test_form_10_config_values(self):
        c = app.get_chunking_config("Form 10")
        self.assertEqual(c["business_chunk_size"], 2500)
        self.assertFalse(c["xbrl_expected"])
        self.assertTrue(c["has_extra_sections"])
        self.assertTrue(c["check_predecessor"])
        self.assertFalse(c["skip_financial_stmts"])

    def test_unknown_form_type_falls_back_to_10k(self):
        self.assertEqual(app.get_chunking_config("8-K"), app.get_chunking_config("10-K"))
        self.assertEqual(app.get_chunking_config(""),    app.get_chunking_config("10-K"))
        self.assertEqual(app.get_chunking_config("XYZ"), app.get_chunking_config("10-K"))


# ══════════════════════════════════════════════════════════════════════════════
# _parse_filename_metadata
# ══════════════════════════════════════════════════════════════════════════════

class TestParseFilenameMetadata(unittest.TestCase):

    def test_standard_10k(self):
        m = app._parse_filename_metadata("AAPL_10K_2024.pdf")
        self.assertEqual(m["ticker"], "AAPL")
        self.assertEqual(m["form_type"], "10-K")
        self.assertEqual(m["fiscal_year"], "2024")
        self.assertIsNone(m["quarter"])

    def test_10q_with_quarter(self):
        m = app._parse_filename_metadata("MSFT_10Q_Q2_2023.pdf")
        self.assertEqual(m["form_type"], "10-Q")
        self.assertEqual(m["quarter"], "Q2")
        self.assertEqual(m["fiscal_year"], "2023")

    def test_lowercase_filename_uppercased(self):
        m = app._parse_filename_metadata("googl_10k_2022.pdf")
        self.assertEqual(m["ticker"], "GOOGL")
        self.assertEqual(m["form_type"], "10-K")

    def test_unknown_file_all_none(self):
        m = app._parse_filename_metadata("document.pdf")
        self.assertIsNone(m["ticker"])
        self.assertIsNone(m["form_type"])
        self.assertIsNone(m["fiscal_year"])
        self.assertIsNone(m["quarter"])

    def test_txt_extension_stripped(self):
        m = app._parse_filename_metadata("TSLA_10K_2023.txt")
        self.assertEqual(m["ticker"], "TSLA")
        self.assertEqual(m["fiscal_year"], "2023")


# ══════════════════════════════════════════════════════════════════════════════
# _detect_fiscal_year_end
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectFiscalYearEnd(unittest.TestCase):

    def test_december_year_end(self):
        text = "For the fiscal year ended December 31, 2024 the company reported..."
        self.assertEqual(app._detect_fiscal_year_end(text), "12-31")

    def test_june_year_end(self):
        text = "For the year ended June 30, 2024 the following results..."
        self.assertEqual(app._detect_fiscal_year_end(text), "06-30")

    def test_march_year_end(self):
        text = "Annual report for the year ended March 31, 2024"
        self.assertEqual(app._detect_fiscal_year_end(text), "03-31")

    def test_no_match_returns_none(self):
        self.assertIsNone(app._detect_fiscal_year_end("No date information in this text."))

    def test_only_checks_first_3000_chars(self):
        # Date buried past the 3 000-char cover page window should be ignored
        text = "X" * 4000 + " fiscal year ended March 31, 2024"
        self.assertIsNone(app._detect_fiscal_year_end(text))


# ══════════════════════════════════════════════════════════════════════════════
# _detect_quarter_from_text
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectQuarterFromText(unittest.TestCase):

    def test_q1_december_fiscal_year(self):
        text = "For the quarter ended March 31, 2024"
        self.assertEqual(app._detect_quarter_from_text(text, "12-31"), "Q1")

    def test_q2_december_fiscal_year(self):
        text = "For the quarter ended June 30, 2024"
        self.assertEqual(app._detect_quarter_from_text(text, "12-31"), "Q2")

    def test_q3_december_fiscal_year(self):
        text = "For the quarter ended September 30, 2024"
        self.assertEqual(app._detect_quarter_from_text(text, "12-31"), "Q3")

    def test_no_period_date_returns_none(self):
        self.assertIsNone(app._detect_quarter_from_text("No period date here.", "12-31"))


# ══════════════════════════════════════════════════════════════════════════════
# _split_into_sections
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_10K = (
    "ITEM 1. BUSINESS\n"
    "We are a technology company providing financial services worldwide.\n\n"
    "ITEM 1A. RISK FACTORS\n"
    "Competition Risk: We face intense competition from established firms.\n\n"
    "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\n"
    "Revenue increased 15% year-over-year driven by core segments.\n\n"
    "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA\n"
    "Consolidated Statements of Operations\nRevenue $1,000,000\n"
)

class TestSplitIntoSections(unittest.TestCase):

    def test_10k_detects_expected_sections(self):
        sections = app._split_into_sections(SAMPLE_10K, "10-K")
        self.assertIn("business", sections)
        self.assertIn("risk_factors", sections)
        self.assertIn("mdna", sections)
        self.assertIn("financial_stmts", sections)

    def test_form_10_label_same_as_internal_10(self):
        s1 = app._split_into_sections(SAMPLE_10K, "Form 10")
        s2 = app._split_into_sections(SAMPLE_10K, "10")
        self.assertEqual(set(s1.keys()), set(s2.keys()))

    def test_no_headers_returns_general(self):
        sections = app._split_into_sections("Plain text with no SEC item headers.", "10-K")
        self.assertIn("general", sections)
        self.assertEqual(len(sections), 1)

    def test_10q_uses_part_i_ii_patterns(self):
        text = (
            "PART I  Item 1. Financial Statements\n"
            "Balance sheet data here.\n\n"
            "PART I  Item 2. Management's Discussion and Analysis\n"
            "Revenue discussion here.\n"
        )
        sections = app._split_into_sections(text, "10-Q")
        self.assertIn("financial_stmts", sections)
        self.assertIn("mdna", sections)

    def test_all_section_texts_non_empty(self):
        sections = app._split_into_sections(SAMPLE_10K, "10-K")
        for name, text in sections.items():
            self.assertTrue(text.strip(), f"Section '{name}' is empty")

    def test_sections_are_non_overlapping(self):
        sections = app._split_into_sections(SAMPLE_10K, "10-K")
        # business section should NOT contain risk factor text
        self.assertNotIn("ITEM 1A", sections.get("business", ""))


# ══════════════════════════════════════════════════════════════════════════════
# _chunk_by_paragraphs
# ══════════════════════════════════════════════════════════════════════════════

class TestChunkByParagraphs(unittest.TestCase):

    def test_short_text_is_single_chunk(self):
        text = "First paragraph.\n\nSecond paragraph."
        # min_length must be below the combined length (36 chars) to get a chunk
        chunks = app._chunk_by_paragraphs(text, chunk_size=1000, overlap=200, min_length=10)
        self.assertEqual(len(chunks), 1)

    def test_large_text_produces_multiple_chunks(self):
        para = "Word " * 100  # ~500 chars each
        text = "\n\n".join([para] * 8)
        chunks = app._chunk_by_paragraphs(text, chunk_size=600, overlap=100)
        self.assertGreater(len(chunks), 1)

    def test_min_length_filters_very_short_paragraphs(self):
        text = "Hi.\n\nThis paragraph is long enough to be kept as a chunk in the output."
        chunks = app._chunk_by_paragraphs(text, chunk_size=500, overlap=100, min_length=20)
        self.assertFalse(any(c.strip() == "Hi." for c in chunks))

    def test_oversized_single_paragraph_split(self):
        big_para = "Word " * 500  # ~2 500 chars
        chunks = app._chunk_by_paragraphs(big_para, chunk_size=500, overlap=100)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 500)

    def test_returns_list_of_strings(self):
        chunks = app._chunk_by_paragraphs("A paragraph here.", chunk_size=500, overlap=100)
        self.assertIsInstance(chunks, list)
        for c in chunks:
            self.assertIsInstance(c, str)


# ══════════════════════════════════════════════════════════════════════════════
# _detect_statement_boundaries
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectStatementBoundaries(unittest.TestCase):

    def test_income_statement_detected(self):
        text = "\nConsolidated Statements of Operations\n\nRevenue 1,000\n"
        types = [b[1] for b in app._detect_statement_boundaries(text)]
        self.assertIn("income_statement", types)

    def test_balance_sheet_detected(self):
        text = "\nConsolidated Balance Sheets\n\nTotal Assets 5,000\n"
        types = [b[1] for b in app._detect_statement_boundaries(text)]
        self.assertIn("balance_sheet", types)

    def test_cash_flow_detected(self):
        text = "\nConsolidated Statements of Cash Flows\n\nOperating 500\n"
        types = [b[1] for b in app._detect_statement_boundaries(text)]
        self.assertIn("cash_flow", types)

    def test_no_statements_returns_empty(self):
        self.assertEqual(app._detect_statement_boundaries("No headers here."), [])

    def test_no_duplicate_statement_types(self):
        # Same header appearing twice — only the first should be captured
        text = (
            "\nConsolidated Statements of Operations\n\nQ1 data\n\n"
            "Consolidated Statements of Operations\n\nQ2 data\n"
        )
        types = [b[1] for b in app._detect_statement_boundaries(text)]
        self.assertEqual(types.count("income_statement"), 1)

    def test_boundaries_sorted_by_position(self):
        text = (
            "\nConsolidated Balance Sheets\n\nAssets 1000\n\n"
            "Consolidated Statements of Operations\n\nRevenue 2000\n"
        )
        boundaries = app._detect_statement_boundaries(text)
        positions = [b[0] for b in boundaries]
        self.assertEqual(positions, sorted(positions))


# ══════════════════════════════════════════════════════════════════════════════
# _scan_audit_status
# ══════════════════════════════════════════════════════════════════════════════

class TestScanAuditStatus(unittest.TestCase):

    def test_unaudited_in_window(self):
        text = "UNAUDITED\nConsolidated Statements of Operations\nRevenue 1000"
        audited, note = app._scan_audit_status(text, "2024-03-31")
        self.assertFalse(audited)
        self.assertIn("unaudited", note)

    def test_predecessor_period(self):
        text = "PREDECESSOR PERIOD\nFor the year ended December 31, 2022\nRevenue 500"
        audited, note = app._scan_audit_status(text, "2022-12-31")
        self.assertTrue(audited)
        self.assertIn("predecessor", note)

    def test_standard_current_period_audited(self):
        text = "Revenue 1000\nCost 600\nSee accompanying notes to the financial statements."
        audited, note = app._scan_audit_status(text, "2024-12-31")
        self.assertTrue(audited)


# ══════════════════════════════════════════════════════════════════════════════
# build_chroma_filter
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildChromaFilter(unittest.TestCase):

    def test_single_field(self):
        f = app.build_chroma_filter({"ticker": "AAPL"})
        self.assertEqual(f, {"ticker": {"$eq": "AAPL"}})

    def test_multi_field_uses_and(self):
        f = app.build_chroma_filter({"ticker": "AAPL", "form_type": "10-K"})
        self.assertIn("$and", f)
        self.assertEqual(len(f["$and"]), 2)

    def test_none_and_empty_values_dropped(self):
        f = app.build_chroma_filter({"ticker": "AAPL", "section": None, "quarter": ""})
        self.assertEqual(f, {"ticker": {"$eq": "AAPL"}})

    def test_all_empty_returns_none(self):
        self.assertIsNone(app.build_chroma_filter({"ticker": None, "quarter": ""}))

    def test_int_value_cast_to_str(self):
        f = app.build_chroma_filter({"fiscal_year": 2024})
        self.assertEqual(f, {"fiscal_year": {"$eq": "2024"}})

    def test_empty_dict_returns_none(self):
        self.assertIsNone(app.build_chroma_filter({}))


# ══════════════════════════════════════════════════════════════════════════════
# route_query
# ══════════════════════════════════════════════════════════════════════════════

class TestRouteQuery(unittest.TestCase):

    def test_revenue_question_is_structured(self):
        self.assertEqual(app.route_query("What was the total revenue for fiscal year 2024?"), "structured")

    def test_net_income_is_structured(self):
        self.assertEqual(app.route_query("What were earnings and net income last quarter?"), "structured")

    def test_why_question_is_narrative(self):
        # Avoid structured signals ("margin", "quarter") — use a clean narrative question
        self.assertEqual(app.route_query("Why did the company face competitive challenges?"), "narrative")

    def test_risk_factors_is_narrative(self):
        self.assertEqual(app.route_query("What are the key risk factors?"), "narrative")

    def test_strategy_is_narrative(self):
        self.assertEqual(app.route_query("Describe the company strategy and outlook."), "narrative")

    def test_returns_valid_value(self):
        for q in ["What is revenue?", "Explain the risks.", "Tell me about the business."]:
            result = app.route_query(q)
            self.assertIn(result, ("structured", "narrative", "both"),
                          f"Unexpected route for: {q!r}")


# ══════════════════════════════════════════════════════════════════════════════
# _fmt_xbrl
# ══════════════════════════════════════════════════════════════════════════════

class TestFmtXbrl(unittest.TestCase):

    def test_trillions(self):
        self.assertEqual(app._fmt_xbrl(2_500_000_000_000, "Revenue"), "$2.50T")

    def test_billions(self):
        self.assertEqual(app._fmt_xbrl(3_800_000_000, "Net Income"), "$3.80B")

    def test_millions(self):
        self.assertEqual(app._fmt_xbrl(450_000_000, "Gross Profit"), "$450.0M")

    def test_small_value(self):
        self.assertEqual(app._fmt_xbrl(1234, "Operating Income"), "$1,234")

    def test_eps_label_uses_dollar_format(self):
        self.assertEqual(app._fmt_xbrl(5.67, "EPS (Diluted)"), "$5.67")

    def test_negative_value(self):
        result = app._fmt_xbrl(-500_000_000, "Net Income")
        self.assertIn("$", result)
        self.assertIn("-", result)


# ══════════════════════════════════════════════════════════════════════════════
# _chunk_all_sections — integration
# ══════════════════════════════════════════════════════════════════════════════

BASE_META = {
    "company": "Test Corp", "ticker": "TEST", "cik": "0000000001",
    "form_type": "10-K", "fiscal_year": "2024", "quarter": "",
    "period_end": "", "source": "test.pdf", "text_source": "pdf_upload",
    "audited": True,
}

class TestChunkAllSections(unittest.TestCase):

    def test_produces_chunks_from_business_section(self):
        sections = {"business": "We provide cloud services to enterprises worldwide. " * 30}
        config = app.get_chunking_config("10-K")
        chunks = app._chunk_all_sections(sections, config, BASE_META, xbrl_available=False)
        self.assertGreater(len(chunks), 0)

    def test_section_name_attached_to_metadata(self):
        # Each risk chunk must be ≥ 150 chars — use long per-risk paragraphs
        risk_text = (
            "CREDIT RISK DISCLOSURES\n"
            "Deterioration in counterparty credit quality may adversely impact "
            "the fair value of our financial instruments held for trading. " * 3
            + "\n\n"
            "MARKET RISK DISCLOSURES\n"
            "Fluctuations in interest rates and foreign exchange rates expose "
            "our investment portfolio to significant valuation uncertainty. " * 3
            + "\n\n"
        )
        sections = {
            "business":     "Cloud services description. " * 30,
            "risk_factors": risk_text * 3,
        }
        config = app.get_chunking_config("10-K")
        chunks = app._chunk_all_sections(sections, config, BASE_META, xbrl_available=False)
        section_values = {c["metadata"]["section"] for c in chunks}
        self.assertIn("business", section_values)
        self.assertIn("risk_factors", section_values)

    def test_no_id_collisions_across_sections(self):
        """chunk_id resets to 0 per section — IDs must include section to stay unique."""
        sections = {
            "business":     "Business description content. " * 50,
            "mdna":         "MD&A revenue discussion content. " * 50,
            "risk_factors": "RISK HEADER ONE\nDetails about risk one.\n\n"
                            "RISK HEADER TWO\nDetails about risk two.\n" * 10,
        }
        config = app.get_chunking_config("10-K")
        chunks = app._chunk_all_sections(sections, config, BASE_META, xbrl_available=False)

        # Replicate the ID generation logic from index_documents (after our fix)
        ids = [
            hashlib.md5(
                f"{c['metadata'].get('source','doc')}__{c['metadata'].get('section','')}"
                f"_chunk_{c['metadata'].get('chunk_id', i)}".encode()
            ).hexdigest()
            for i, c in enumerate(chunks)
        ]
        self.assertEqual(len(ids), len(set(ids)), "ID collision across sections!")

    def test_financial_stmts_skipped_when_xbrl_available_10k(self):
        """10-K with XBRL available: financial_stmts should produce 0 chunks."""
        sections = {
            "financial_stmts": (
                "\nConsolidated Statements of Operations\n"
                "Revenue 1,000,000\nNet Income 200,000\n" * 10
            )
        }
        config = app.get_chunking_config("10-K")  # skip_financial_stmts=True
        chunks = app._chunk_all_sections(sections, config, BASE_META, xbrl_available=True)
        self.assertEqual(len(chunks), 0)

    def test_financial_stmts_indexed_when_form_10_no_xbrl(self):
        """Form 10 has no XBRL; financial_stmts must be chunked."""
        sections = {
            "financial_stmts": (
                "\nConsolidated Statements of Operations\n"
                "Revenue: 1,000,000\nCost: 600,000\nNet Income: 200,000\n" * 20
            )
        }
        config = app.get_chunking_config("Form 10")  # skip_financial_stmts=False
        meta = {**BASE_META, "form_type": "Form 10"}
        chunks = app._chunk_all_sections(sections, config, meta, xbrl_available=False)
        self.assertGreater(len(chunks), 0)

    def test_unknown_section_uses_default_chunker(self):
        sections = {"quantitative": "Interest rate risk discussion. " * 30}
        config = app.get_chunking_config("10-K")
        chunks = app._chunk_all_sections(sections, config, BASE_META, xbrl_available=False)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(c["metadata"]["section"] == "quantitative" for c in chunks))

    def test_empty_section_skipped(self):
        sections = {"business": "   ", "mdna": "Revenue grew significantly. " * 30}
        config = app.get_chunking_config("10-K")
        chunks = app._chunk_all_sections(sections, config, BASE_META, xbrl_available=False)
        section_values = {c["metadata"]["section"] for c in chunks}
        self.assertNotIn("business", section_values)
        self.assertIn("mdna", section_values)


if __name__ == "__main__":
    unittest.main(verbosity=2)

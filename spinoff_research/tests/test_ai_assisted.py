import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from spinoff_research.db import init_db
from spinoff_research.form10_retrieval import Form10NotFoundError, RelevantSection
from spinoff_research.models import Company, SpinoffTransaction
from spinoff_research.repository import get_or_create_transaction
from spinoff_research.status import FieldStatus


def _mock_claude_response(payload: dict):
    """Builds a MagicMock shaped like anthropic.types.Message — a
    single text content block whose .text is the JSON-schema-constrained
    output extract_ceo_came_from_parent expects."""
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps(payload)
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    return response


class TestExtractCeoCameFromParent(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.conn = init_db(Path(self._tmpdir.name) / "test.db")
        tx = SpinoffTransaction(
            parent=Company(name="Parent Co", ticker="P", cik="0000000001"),
            spinoff=Company(name="Spinco Inc", ticker="S", cik="0000000002"),
            announcement_date="2024-01-01", spinoff_date="2024-04-01",
        )
        self.txn = get_or_create_transaction(self.conn, tx)

    def tearDown(self):
        self.conn.close()
        self._tmpdir.cleanup()

    @patch("spinoff_research.extractors.ai_assisted.get_or_ingest_form10")
    def test_no_form10_returns_not_found(self, mock_get_form10):
        from spinoff_research.extractors.ai_assisted import extract_ceo_came_from_parent

        mock_get_form10.side_effect = Form10NotFoundError("No Form 10 found for CIK 0000000002")
        result = extract_ceo_came_from_parent(self.conn, self.txn.transaction_id, "0000000002", "Spinco Inc")
        self.assertEqual(result.status, FieldStatus.NOT_FOUND)
        self.assertEqual(result.field_key, "ceo_came_from_parent")
        self.assertEqual(result.model_used, "claude-haiku-4-5")

    @patch("spinoff_research.extractors.ai_assisted.find_relevant_sections")
    @patch("spinoff_research.extractors.ai_assisted.get_or_ingest_form10")
    def test_no_relevant_sections_returns_not_yet_determinable(self, mock_get_form10, mock_find):
        from spinoff_research.extractors.ai_assisted import extract_ceo_came_from_parent

        mock_get_form10.return_value = [42]
        mock_find.return_value = []
        result = extract_ceo_came_from_parent(self.conn, self.txn.transaction_id, "0000000002", "Spinco Inc")
        self.assertEqual(result.status, FieldStatus.NOT_YET_DETERMINABLE)

    @patch("spinoff_research.extractors.ai_assisted._call_claude")
    @patch("spinoff_research.extractors.ai_assisted.find_relevant_sections")
    @patch("spinoff_research.extractors.ai_assisted.get_or_ingest_form10")
    def test_true_answer_with_verified_excerpt_is_high_confidence(self, mock_get_form10, mock_find, mock_call):
        from spinoff_research.extractors.ai_assisted import extract_ceo_came_from_parent

        mock_get_form10.return_value = [42]
        excerpt = "Jane Doe has served as our Chief Executive Officer since 2020. Prior to the separation, she was Executive Vice President at Parent Co."
        mock_find.return_value = [RelevantSection(section_id=7, document_id=42, content=excerpt)]
        mock_call.return_value = {
            "answer": "true",
            "supporting_excerpt": "Prior to the separation, she was Executive Vice President at Parent Co.",
            "reasoning_summary": "The filing states she held a role at the parent before separation.",
        }

        result = extract_ceo_came_from_parent(self.conn, self.txn.transaction_id, "0000000002", "Spinco Inc")
        self.assertEqual(result.status, FieldStatus.EXTRACTED_HIGH_CONFIDENCE)
        self.assertEqual(result.raw_value, "True")
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].section_id, 7)
        self.assertIn("Executive Vice President", result.sources[0].supporting_excerpt)

    @patch("spinoff_research.extractors.ai_assisted._call_claude")
    @patch("spinoff_research.extractors.ai_assisted.find_relevant_sections")
    @patch("spinoff_research.extractors.ai_assisted.get_or_ingest_form10")
    def test_smart_quote_punctuation_difference_still_verifies(self, mock_get_form10, mock_find, mock_call):
        """Regression for a real live-validation finding (GE Vernova,
        Inhibrx pilots): source HTML uses curly quotes ("CEO") but Claude's
        JSON-echoed 'verbatim' excerpt normalizes them to straight quotes
        ("CEO") — a typography difference, not a hallucination, and must
        not be flagged EXTRACTED_UNCERTAIN."""
        from spinoff_research.extractors.ai_assisted import extract_ceo_came_from_parent

        source_text = (
            'Mr. Strazik has been the Chief Executive Officer (“CEO”) of the '
            'business since November 2021 and previously served as Chief Financial '
            'Officer of the parent company’s power division.'
        )
        mock_get_form10.return_value = [42]
        mock_find.return_value = [RelevantSection(section_id=7, document_id=42, content=source_text)]
        mock_call.return_value = {
            "answer": "true",
            "supporting_excerpt": 'Mr. Strazik has been the Chief Executive Officer ("CEO") of the '
                                   'business since November 2021 and previously served as Chief Financial '
                                   "Officer of the parent company's power division.",
            "reasoning_summary": "He held a CFO role at the parent before becoming CEO.",
        }

        result = extract_ceo_came_from_parent(self.conn, self.txn.transaction_id, "0000000002", "Spinco Inc")
        self.assertEqual(result.status, FieldStatus.EXTRACTED_HIGH_CONFIDENCE)
        self.assertNotIn("WARNING", result.sources[0].reasoning_summary)

    @patch("spinoff_research.extractors.ai_assisted._call_claude")
    @patch("spinoff_research.extractors.ai_assisted.find_relevant_sections")
    @patch("spinoff_research.extractors.ai_assisted.get_or_ingest_form10")
    def test_false_answer_with_verified_excerpt(self, mock_get_form10, mock_find, mock_call):
        from spinoff_research.extractors.ai_assisted import extract_ceo_came_from_parent

        mock_get_form10.return_value = [42]
        excerpt = "John Smith joined the company as Chief Executive Officer in March 2024, having previously served as CEO of an unrelated public company."
        mock_find.return_value = [RelevantSection(section_id=8, document_id=42, content=excerpt)]
        mock_call.return_value = {
            "answer": "false",
            "supporting_excerpt": "John Smith joined the company as Chief Executive Officer in March 2024, having previously served as CEO of an unrelated public company.",
            "reasoning_summary": "He was hired externally, not from the parent.",
        }

        result = extract_ceo_came_from_parent(self.conn, self.txn.transaction_id, "0000000002", "Spinco Inc")
        self.assertEqual(result.status, FieldStatus.EXTRACTED_HIGH_CONFIDENCE)
        self.assertEqual(result.raw_value, "False")

    @patch("spinoff_research.extractors.ai_assisted._call_claude")
    @patch("spinoff_research.extractors.ai_assisted.find_relevant_sections")
    @patch("spinoff_research.extractors.ai_assisted.get_or_ingest_form10")
    def test_not_stated_answer_returns_not_yet_determinable(self, mock_get_form10, mock_find, mock_call):
        from spinoff_research.extractors.ai_assisted import extract_ceo_came_from_parent

        mock_get_form10.return_value = [42]
        mock_find.return_value = [RelevantSection(section_id=9, document_id=42, content="unrelated text about capital structure")]
        mock_call.return_value = {"answer": "not_stated", "supporting_excerpt": "", "reasoning_summary": "No mention of prior employment."}

        result = extract_ceo_came_from_parent(self.conn, self.txn.transaction_id, "0000000002", "Spinco Inc")
        self.assertEqual(result.status, FieldStatus.NOT_YET_DETERMINABLE)
        self.assertIsNone(result.raw_value)

    @patch("spinoff_research.extractors.ai_assisted._call_claude")
    @patch("spinoff_research.extractors.ai_assisted.find_relevant_sections")
    @patch("spinoff_research.extractors.ai_assisted.get_or_ingest_form10")
    def test_hallucinated_excerpt_not_in_source_is_flagged_uncertain(self, mock_get_form10, mock_find, mock_call):
        """The core faithfulness guard: if Claude's cited excerpt is not an
        exact substring of what we actually gave it, don't trust the
        citation — mark it uncertain for human review rather than silently
        accepting a fabricated quote."""
        from spinoff_research.extractors.ai_assisted import extract_ceo_came_from_parent

        mock_get_form10.return_value = [42]
        mock_find.return_value = [RelevantSection(section_id=10, document_id=42, content="Jane Doe has served as CEO since 2020.")]
        mock_call.return_value = {
            "answer": "true",
            "supporting_excerpt": "This exact sentence does not appear anywhere in the source text.",
            "reasoning_summary": "Claimed prior parent role.",
        }

        result = extract_ceo_came_from_parent(self.conn, self.txn.transaction_id, "0000000002", "Spinco Inc")
        self.assertEqual(result.status, FieldStatus.EXTRACTED_UNCERTAIN)
        self.assertLess(result.confidence, 0.5)
        self.assertIn("WARNING", result.sources[0].reasoning_summary)

    @patch("spinoff_research.extractors.ai_assisted._call_claude")
    @patch("spinoff_research.extractors.ai_assisted.find_relevant_sections")
    @patch("spinoff_research.extractors.ai_assisted.get_or_ingest_form10")
    def test_claude_call_failure_requires_manual_review(self, mock_get_form10, mock_find, mock_call):
        import anthropic
        from spinoff_research.extractors.ai_assisted import extract_ceo_came_from_parent

        mock_get_form10.return_value = [42]
        mock_find.return_value = [RelevantSection(section_id=11, document_id=42, content="some bio text")]
        mock_call.side_effect = anthropic.APIConnectionError(request=MagicMock())

        result = extract_ceo_came_from_parent(self.conn, self.txn.transaction_id, "0000000002", "Spinco Inc")
        self.assertEqual(result.status, FieldStatus.REQUIRES_MANUAL_REVIEW)


class TestCallClaudeCostLogging(unittest.TestCase):
    """_call_claude itself (not mocked away) — verifies every real API
    call logs its cost, using the same $ get a sense of how much this is
    costing ask that drove building cost_tracking.py in the first place."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._log_path = Path(self._tmpdir.name) / "ai_cost_log.jsonl"
        self._log_patcher = patch("spinoff_research.cost_tracking._LOG_PATH", self._log_path)
        self._log_patcher.start()

    def tearDown(self):
        self._log_patcher.stop()
        self._tmpdir.cleanup()

    @patch("anthropic.Anthropic")
    def test_successful_call_logs_cost(self, mock_anthropic_cls):
        from spinoff_research.cost_tracking import read_log
        from spinoff_research.extractors.ai_assisted import _call_claude

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        response = _mock_claude_response({"answer": "true", "supporting_excerpt": "x", "reasoning_summary": "y"})
        response.usage.input_tokens = 8000
        response.usage.output_tokens = 150
        mock_client.messages.create.return_value = response

        _call_claude("Spinco Inc", "some excerpt text", "ceo_came_from_parent")

        entries = read_log()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["input_tokens"], 8000)
        self.assertEqual(entries[0]["output_tokens"], 150)
        self.assertGreater(entries[0]["cost_usd"], 0)

    @patch("anthropic.Anthropic")
    def test_unparseable_response_still_logs_cost(self, mock_anthropic_cls):
        """A response with no text block still consumed real tokens and
        must still be billed in the log, even though the caller goes on
        to raise AiExtractionError."""
        from spinoff_research.cost_tracking import read_log
        from spinoff_research.extractors.ai_assisted import _call_claude, AiExtractionError

        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        response = MagicMock()
        response.content = []
        response.stop_reason = "max_tokens"
        response.usage.input_tokens = 500
        response.usage.output_tokens = 400
        mock_client.messages.create.return_value = response

        with self.assertRaises(AiExtractionError):
            _call_claude("Spinco Inc", "some excerpt text", "ceo_came_from_parent")

        entries = read_log()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["input_tokens"], 500)


if __name__ == "__main__":
    unittest.main()

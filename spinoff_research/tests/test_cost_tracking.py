import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestCostTracking(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._log_path = Path(self._tmpdir.name) / "ai_cost_log.jsonl"
        self._patcher = patch("spinoff_research.cost_tracking._LOG_PATH", self._log_path)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._tmpdir.cleanup()

    def test_compute_cost_usd_matches_haiku_pricing(self):
        from spinoff_research.cost_tracking import compute_cost_usd
        # 1M input tokens @ $1.00, 1M output tokens @ $5.00
        cost = compute_cost_usd("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(cost, 6.00, places=6)

    def test_compute_cost_usd_unknown_model_raises(self):
        from spinoff_research.cost_tracking import compute_cost_usd
        with self.assertRaises(KeyError):
            compute_cost_usd("gpt-4o", input_tokens=100, output_tokens=100)

    def test_log_ai_call_appends_entry_and_returns_cost(self):
        from spinoff_research.cost_tracking import log_ai_call, read_log

        cost = log_ai_call(
            model="claude-haiku-4-5", field_key="ceo_came_from_parent",
            caller="Spinco Inc", input_tokens=8000, output_tokens=200,
        )
        self.assertGreater(cost, 0)

        entries = read_log()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["field_key"], "ceo_came_from_parent")
        self.assertEqual(entries[0]["caller"], "Spinco Inc")
        self.assertEqual(entries[0]["input_tokens"], 8000)
        self.assertEqual(entries[0]["output_tokens"], 200)

    def test_read_log_returns_empty_list_when_no_log_exists(self):
        from spinoff_research.cost_tracking import read_log
        self.assertEqual(read_log(), [])

    def test_read_log_since_filters_by_timestamp(self):
        from spinoff_research.cost_tracking import log_ai_call, read_log

        old_entry = {
            "timestamp": "2020-01-01T00:00:00", "model": "claude-haiku-4-5",
            "field_key": "x", "caller": "old", "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.000001,
        }
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("w") as f:
            f.write(json.dumps(old_entry) + "\n")

        log_ai_call(model="claude-haiku-4-5", field_key="y", caller="new", input_tokens=100, output_tokens=100)

        all_entries = read_log()
        self.assertEqual(len(all_entries), 2)

        recent_only = read_log(since="2026-01-01T00:00:00")
        self.assertEqual(len(recent_only), 1)
        self.assertEqual(recent_only[0]["caller"], "new")

    def test_total_cost_usd_sums_all_entries(self):
        from spinoff_research.cost_tracking import log_ai_call, total_cost_usd

        log_ai_call(model="claude-haiku-4-5", field_key="a", caller="c1", input_tokens=1_000_000, output_tokens=0)
        log_ai_call(model="claude-haiku-4-5", field_key="b", caller="c2", input_tokens=1_000_000, output_tokens=0)

        self.assertAlmostEqual(total_cost_usd(), 2.00, places=6)


if __name__ == "__main__":
    unittest.main()

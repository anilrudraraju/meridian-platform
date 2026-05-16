"""
core/cost.py — daily spend cap + per-call logging for all LLM calls.
"""
import os
import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional

COST_LOG_PATH  = Path("/tmp/meridian_cost_log.jsonl")
DAILY_CAP_USD  = float(os.environ.get("MERIDIAN_DAILY_CAP", "5.00"))


def _today() -> str:
    return date.today().isoformat()


def daily_spend() -> float:
    """Sum all logged costs for today."""
    if not COST_LOG_PATH.exists():
        return 0.0
    total = 0.0
    today = _today()
    try:
        with COST_LOG_PATH.open() as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("date") == today:
                    total += entry.get("cost_usd", 0.0)
    except Exception:
        pass
    return total


def check_budget() -> tuple[bool, float, float]:
    """
    Returns (under_budget, spent_today, cap).
    Callers should abort the LLM call when under_budget is False.
    """
    spent = daily_spend()
    return spent < DAILY_CAP_USD, spent, DAILY_CAP_USD


def log_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    technique: Optional[str] = None,
    caller: Optional[str] = None,
) -> None:
    """Append one record to the cost log."""
    entry = {
        "date":              _today(),
        "timestamp":         datetime.now().isoformat(),
        "model":             model,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens":      prompt_tokens + completion_tokens,
        "cost_usd":          round(cost_usd, 6),
        "technique":         technique or "",
        "caller":            caller or "",
    }
    try:
        with COST_LOG_PATH.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def read_log() -> list[dict]:
    """Return all entries from the cost log, newest first."""
    if not COST_LOG_PATH.exists():
        return []
    entries = []
    try:
        with COST_LOG_PATH.open() as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    except Exception:
        pass
    return list(reversed(entries))

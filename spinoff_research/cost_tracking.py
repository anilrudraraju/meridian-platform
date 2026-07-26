"""
Cost tracking for AI-assisted CLI extraction (spinoff_research/extractors/
ai_assisted.py). Deliberately separate from core/cost.py — that module logs
to /tmp (fine for the Streamlit app, which the CLI doesn't run alongside)
and tracks Meridian's own daily budget cap, a different concern from "how
much has this research pipeline spent on Claude calls, ever."

Log lives next to the SQLite DB (spinoff_research/data/, already gitignored
as proprietary/local-only) so it survives across CLI invocations, unlike
/tmp on most systems.
"""
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_LOG_PATH = Path(__file__).parent / "data" / "ai_cost_log.jsonl"

# $ per 1M tokens. Update if pricing changes or a new model is used here.
_PRICING_PER_MILLION = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


@dataclass
class CostLogEntry:
    timestamp: str
    model: str
    field_key: str
    caller: str            # e.g. transaction label, for readability in the log
    input_tokens: int
    output_tokens: int
    cost_usd: float


def compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Raises KeyError for an unrecognized model — a silent $0.00 for a
    real call would be a worse failure than a loud one at development
    time, since this number is what the user relies on for budgeting."""
    rates = _PRICING_PER_MILLION[model]
    return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000


def log_ai_call(model: str, field_key: str, caller: str, input_tokens: int, output_tokens: int) -> float:
    """Appends one entry and returns the computed cost in USD for this call."""
    cost_usd = compute_cost_usd(model, input_tokens, output_tokens)
    entry = CostLogEntry(
        timestamp=datetime.now().isoformat(), model=model, field_key=field_key,
        caller=caller, input_tokens=input_tokens, output_tokens=output_tokens,
        cost_usd=round(cost_usd, 6),
    )
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a") as f:
        f.write(json.dumps(entry.__dict__) + "\n")
    return cost_usd


def read_log(since: Optional[str] = None) -> List[dict]:
    """All logged entries, oldest first. since: ISO timestamp string —
    only entries at or after this time, for scoping a single CLI run's
    total rather than all-time spend."""
    if not _LOG_PATH.exists():
        return []
    entries = []
    with _LOG_PATH.open() as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if since is None or entry["timestamp"] >= since:
                entries.append(entry)
    return entries


def total_cost_usd(since: Optional[str] = None) -> float:
    return sum(e["cost_usd"] for e in read_log(since=since))

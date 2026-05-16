import csv
import os
from typing import List, Dict
from spinoff.models import CostEntry

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__),
                             "..", "data", "spinoffs", "sample_cost_log.csv")
_FIELDS = ["date", "category", "description", "amount_usd", "ticker"]
CATEGORIES = ["OpenAI", "Data", "Research", "Other"]


def load_costs(path: str = _DEFAULT_PATH) -> List[CostEntry]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            CostEntry(
                date=row["date"],
                category=row["category"],
                description=row["description"],
                amount_usd=float(row["amount_usd"]),
                ticker=row.get("ticker", ""),
            )
            for row in reader
        ]


def save_costs(costs: List[CostEntry], path: str = _DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for c in costs:
            writer.writerow({
                "date": c.date,
                "category": c.category,
                "description": c.description,
                "amount_usd": c.amount_usd,
                "ticker": c.ticker,
            })


def log_cost(entry: CostEntry, path: str = _DEFAULT_PATH) -> None:
    existing = load_costs(path)
    existing.append(entry)
    save_costs(existing, path)


def total_by_category(costs: List[CostEntry]) -> Dict[str, float]:
    totals: Dict[str, float] = {cat: 0.0 for cat in CATEGORIES}
    for c in costs:
        key = c.category if c.category in totals else "Other"
        totals[key] += c.amount_usd
    return totals

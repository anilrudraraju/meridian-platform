import csv
import os
from typing import List
from spinoff.models import ThesisEntry

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__),
                             "..", "data", "spinoffs", "sample_thesis_tracker.csv")
_FIELDS = ["ticker", "hypothesis", "entry_date", "target_price", "catalyst",
           "current_thesis", "notes"]


def load_theses(path: str = _DEFAULT_PATH) -> List[ThesisEntry]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            ThesisEntry(
                ticker=row["ticker"],
                hypothesis=row["hypothesis"],
                entry_date=row["entry_date"],
                target_price=float(row["target_price"]),
                catalyst=row["catalyst"],
                current_thesis=row.get("current_thesis", "Active"),
                notes=row.get("notes", ""),
            )
            for row in reader
        ]


def save_theses(theses: List[ThesisEntry], path: str = _DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for t in theses:
            writer.writerow({
                "ticker": t.ticker,
                "hypothesis": t.hypothesis,
                "entry_date": t.entry_date,
                "target_price": t.target_price,
                "catalyst": t.catalyst,
                "current_thesis": t.current_thesis,
                "notes": t.notes,
            })


def add_thesis(entry: ThesisEntry, path: str = _DEFAULT_PATH) -> None:
    existing = load_theses(path)
    existing.append(entry)
    save_theses(existing, path)

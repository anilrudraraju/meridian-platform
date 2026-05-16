import csv
import os
from typing import List
from spinoff.models import ManagementPromise

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__),
                             "..", "data", "spinoffs", "sample_management_promises.csv")
_FIELDS = ["ticker", "promise_date", "promise", "source", "status", "notes"]


def load_promises(path: str = _DEFAULT_PATH) -> List[ManagementPromise]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            ManagementPromise(
                ticker=row["ticker"],
                promise_date=row["promise_date"],
                promise=row["promise"],
                source=row["source"],
                status=row.get("status", "Pending"),
                notes=row.get("notes", ""),
            )
            for row in reader
        ]


def save_promises(promises: List[ManagementPromise], path: str = _DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        for p in promises:
            writer.writerow({
                "ticker": p.ticker,
                "promise_date": p.promise_date,
                "promise": p.promise,
                "source": p.source,
                "status": p.status,
                "notes": p.notes,
            })


def add_promise(promise: ManagementPromise, path: str = _DEFAULT_PATH) -> None:
    existing = load_promises(path)
    existing.append(promise)
    save_promises(existing, path)

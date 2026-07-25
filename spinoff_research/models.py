"""
Company and transaction domain models for the spin-off research prototype.

These mirror the companies/spinoff_transactions tables in schema.sql but
exist as plain dataclasses so sec_service.py and the loaders can pass
typed values around without a live DB connection open. Persisting an
instance is a separate, explicit step (see db.py table shapes) — these
are not ORM-bound records.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Company:
    name: str
    ticker: Optional[str] = None
    cik: Optional[str] = None          # zero-padded 10-digit SEC CIK, once resolved
    sector: Optional[str] = None
    industry: Optional[str] = None
    company_id: Optional[int] = None   # set once persisted to the DB

    def display_name(self) -> str:
        return f"{self.name} ({self.ticker})" if self.ticker else self.name


@dataclass
class SpinoffTransaction:
    parent: Company
    spinoff: Company
    announcement_date: Optional[str] = None  # ISO date, once known
    spinoff_date: Optional[str] = None      # ISO date, once known (regular-way trading date)
    source_row_number: Optional[int] = None  # traceability back to Rich's sheet row
    status: str = "pilot"                    # 'pilot' | 'active' | 'archived'
    transaction_id: Optional[int] = None     # set once persisted to the DB

    def label(self) -> str:
        return f"{self.spinoff.display_name()} spun off from {self.parent.display_name()}"

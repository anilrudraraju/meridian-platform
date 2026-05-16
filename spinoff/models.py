from dataclasses import dataclass, field
from typing import List


@dataclass
class SpinoffEvent:
    ticker: str
    parent_ticker: str
    company_name: str
    parent_name: str
    spinoff_date: str       # ISO date "YYYY-MM-DD"
    exchange: str
    sector: str
    cik: str = ""
    notes: str = ""


@dataclass
class GreenblattScore:
    """
    Greenblatt spinoff screening scorecard.
    Each criterion is 0–5; max total = 35.
    Tier: ≥35 Deep Dive | 25–34 Watchlist | <25 Pass
    """
    ticker: str
    company_name: str
    forced_selling: int = 0       # index funds / parent shareholders dumping shares
    insider_incentives: int = 0   # mgmt owns stock / options in the spinoff, not parent
    valuation: int = 0            # EV/EBIT vs peers; hidden asset value
    return_on_capital: int = 0    # standalone ROIC or EBIT margin
    debt_risk: int = 0            # leverage + interest coverage (5=low risk)
    business_quality: int = 0     # moat, recurring revenue, customer concentration
    catalyst_strength: int = 0    # timeline clarity, buyer universe, re-rating path
    notes: str = ""

    @property
    def total(self) -> int:
        return (
            self.forced_selling + self.insider_incentives + self.valuation
            + self.return_on_capital + self.debt_risk
            + self.business_quality + self.catalyst_strength
        )

    @property
    def tier(self) -> str:
        t = self.total
        if t >= 35:
            return "Deep Dive"
        if t >= 25:
            return "Watchlist"
        return "Pass"


@dataclass
class ManagementPromise:
    ticker: str
    promise_date: str       # ISO date "YYYY-MM-DD"
    promise: str
    source: str             # "10-K", "earnings call", "investor day", "press release"
    status: str = "Pending" # "Pending" | "Kept" | "Broken" | "Partially Met"
    notes: str = ""


@dataclass
class ThesisEntry:
    ticker: str
    hypothesis: str
    entry_date: str         # ISO date "YYYY-MM-DD"
    target_price: float
    catalyst: str
    current_thesis: str = "Active"  # "Active" | "Broken" | "Realized"
    notes: str = ""


@dataclass
class CostEntry:
    date: str               # ISO date "YYYY-MM-DD"
    category: str           # "OpenAI" | "Data" | "Research" | "Other"
    description: str
    amount_usd: float
    ticker: str = ""

"""
Greenblatt spinoff screening logic.

Each of 7 criteria is scored 0–5 by the analyst.
Total 0–35:  ≥35 → Deep Dive  |  25–34 → Watchlist  |  <25 → Pass
"""
from spinoff.models import GreenblattScore

CRITERIA = [
    ("forced_selling",     "Forced Selling",      "Index funds and parent shareholders are likely to dump shares post-spin."),
    ("insider_incentives", "Insider Incentives",   "Management owns stock/options in the spinco, not just the parent."),
    ("valuation",          "Valuation / EV-EBIT",  "Attractive EV/EBIT vs peers; possible hidden asset value."),
    ("return_on_capital",  "Return on Capital",    "High standalone ROIC or EBIT margin relative to sector."),
    ("debt_risk",          "Debt Risk",            "Low leverage and strong interest coverage (5 = safest)."),
    ("business_quality",   "Business Quality",     "Moat, recurring revenue, manageable customer concentration."),
    ("catalyst_strength",  "Catalyst Strength",    "Clear re-rating catalyst with identifiable buyer universe."),
]

TIER_THRESHOLDS = {"Deep Dive": 35, "Watchlist": 25}


def tier_label(total: int) -> str:
    if total >= TIER_THRESHOLDS["Deep Dive"]:
        return "Deep Dive"
    if total >= TIER_THRESHOLDS["Watchlist"]:
        return "Watchlist"
    return "Pass"


def score_spinoff(
    ticker: str,
    company_name: str,
    forced_selling: int,
    insider_incentives: int,
    valuation: int,
    return_on_capital: int,
    debt_risk: int,
    business_quality: int,
    catalyst_strength: int,
    notes: str = "",
) -> GreenblattScore:
    """Validate scores and return a GreenblattScore dataclass."""
    scores = {
        "forced_selling": forced_selling,
        "insider_incentives": insider_incentives,
        "valuation": valuation,
        "return_on_capital": return_on_capital,
        "debt_risk": debt_risk,
        "business_quality": business_quality,
        "catalyst_strength": catalyst_strength,
    }
    for k, v in scores.items():
        if not (0 <= v <= 5):
            raise ValueError(f"{k} must be 0–5, got {v}")

    return GreenblattScore(
        ticker=ticker,
        company_name=company_name,
        notes=notes,
        **scores,
    )

"""
Synthetic SEC XBRL companyfacts fixtures for testing xbrl_service.py.
Shapes mirror real cases confirmed live against GE Vernova, Inhibrx, and
Grail during development — not arbitrary made-up data.
"""

# Two concepts agree exactly for the same (period_start=None, period_end) —
# the "clean" case: dei cover-page fact and us-gaap balance-sheet fact
# happen to both report 2024-04-02 with the same value (as seen for GEV).
FACTS_SHARES_AGREE = {
    "facts": {
        "dei": {
            "EntityCommonStockSharesOutstanding": {
                "units": {"shares": [
                    {"end": "2024-04-02", "val": 274085523, "accn": "0001-24-000001", "form": "10-Q", "filed": "2024-04-30"},
                ]}
            }
        },
        "us-gaap": {
            "CommonStockSharesOutstanding": {
                "units": {"shares": [
                    {"end": "2024-04-02", "val": 274085523, "accn": "0001-24-000002", "form": "10-Q", "filed": "2024-07-24"},
                ]}
            }
        },
    }
}

# Two concepts report DIFFERENT values for the exact same period — a real conflict.
FACTS_DEBT_CONFLICT = {
    "facts": {
        "us-gaap": {
            "LongTermDebt": {
                "units": {"USD": [
                    {"end": "2024-06-30", "val": 500000000, "accn": "0001-24-000010", "form": "10-Q", "filed": "2024-08-01"},
                ]}
            },
            "LongTermDebtNoncurrent": {
                "units": {"USD": [
                    {"end": "2024-06-30", "val": 480000000, "accn": "0001-24-000011", "form": "10-Q", "filed": "2024-08-01"},
                ]}
            },
        }
    }
}

# Same concept, same period_end, DIFFERENT period_start — quarterly vs. YTD.
# This must NOT be flagged as a conflict (regression fixture for the bug
# found live against Grail's revenue: $31.97M Q2-only vs $58.69M 6-month YTD,
# both ending 2024-06-30).
FACTS_REVENUE_DURATION_FALSE_CONFLICT = {
    "facts": {
        "us-gaap": {
            "RevenueFromContractWithCustomerExcludingAssessedTax": {
                "units": {"USD": [
                    {"start": "2024-04-01", "end": "2024-06-30", "val": 31970000, "accn": "0001-24-000020", "form": "10-Q", "filed": "2024-08-01"},
                    {"start": "2024-01-01", "end": "2024-06-30", "val": 58691000, "accn": "0001-24-000020", "form": "10-Q", "filed": "2024-08-01"},
                    # a genuine annual figure, for the prior fiscal year, ending well before the above
                    {"start": "2023-01-01", "end": "2023-12-31", "val": 120000000, "accn": "0001-24-000005", "form": "10-K", "filed": "2024-02-15"},
                ]}
            }
        }
    }
}

# A snapshot field with values at multiple points in time — the "current
# value drift" case (GEV shares outstanding: ~274M at distribution,
# ~266M "today" many quarters later).
FACTS_SHARES_DRIFT_OVER_TIME = {
    "facts": {
        "us-gaap": {
            "CommonStockSharesOutstanding": {
                "units": {"shares": [
                    {"end": "2024-04-02", "val": 274085523, "accn": "0001-24-000001", "form": "10-Q", "filed": "2024-04-30"},
                    {"end": "2025-06-30", "val": 270000000, "accn": "0001-25-000050", "form": "10-Q", "filed": "2025-07-24"},
                    {"end": "2026-06-30", "val": 266333581, "accn": "0001-26-000090", "form": "10-Q", "filed": "2026-07-22"},
                ]}
            }
        }
    }
}

# A pre-spin snapshot reports a literal 0 (entity has no publicly issued
# shares yet) closer in days to the anchor than a real post-spin value —
# confirmed live against Grail: 2024-06-23 (2 days before distribution)
# reports 0 shares; 2024-06-30 (5 days after) reports the real
# post-distribution count of 31,049,148. Distance-only tie-breaking picks
# the closer-but-meaningless 0.
FACTS_SHARES_ZERO_PRE_SPIN = {
    "facts": {
        "us-gaap": {
            "CommonStockSharesOutstanding": {
                "units": {"shares": [
                    {"end": "2024-06-23", "val": 0, "accn": "0001-24-000030", "form": "10-Q", "filed": "2024-08-13"},
                    {"end": "2024-06-30", "val": 31049148, "accn": "0001-24-000031", "form": "10-Q", "filed": "2024-08-13"},
                ]}
            }
        }
    }
}

# Every in-tolerance candidate is genuinely 0 — the zero-avoidance
# preference must not mask this by reaching outside the tolerance window
# or otherwise fabricating a nonzero answer.
FACTS_SHARES_ALL_ZERO = {
    "facts": {
        "us-gaap": {
            "CommonStockSharesOutstanding": {
                "units": {"shares": [
                    {"end": "2024-06-23", "val": 0, "accn": "0001-24-000030", "form": "10-Q", "filed": "2024-08-13"},
                    {"end": "2024-06-24", "val": 0, "accn": "0001-24-000032", "form": "10-Q", "filed": "2024-08-13"},
                ]}
            }
        }
    }
}

# No candidates at all for a field/concept combination.
FACTS_EMPTY = {"facts": {"us-gaap": {}}}

# Only a non-standard/company-extension tag reports the concept.
FACTS_EXTENSION_TAG_ONLY = {
    "facts": {
        "gev": {  # company-specific taxonomy prefix, not a standard SEC taxonomy
            "CustomDebtMeasure": {
                "units": {"USD": [
                    {"end": "2024-04-02", "val": 100000000, "accn": "0001-24-000030", "form": "10-Q", "filed": "2024-04-30"},
                ]}
            }
        }
    }
}

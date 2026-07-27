"""
SEC XBRL structured-fact retrieval and candidate evaluation.

Does NOT reuse core/edgar.py's fetch_xbrl_facts() — that function picks the
first matching tag from a single hardcoded priority list and returns only
the winner, silently. That is exactly the failure mode the brief warns
against: confirmed live against GE Vernova that (a) two different tags
compete for "shares outstanding" (dei:EntityCommonStockSharesOutstanding,
a cover-page/as-of-date figure, vs. us-gaap:CommonStockSharesOutstanding, a
balance-sheet-date figure) and (b) GEV's debt is tagged
LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities, not the
plain LongTermDebt tag core/edgar.py looks for at all. A single-tag lookup
would either silently pick the "wrong" one or return nothing.

This module always resolves a field to a *set* of candidate facts, applies
an explicit, documented selection rule, and keeps the rejected candidates
attached to the result rather than discarding them — so a reviewer (or a
later automation-assessment pass) can see what was rejected and why.
"""
import os
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional

from spinoff_research.sec_service import _paced_get, SecServiceError


@dataclass
class XbrlCandidateFact:
    concept: str                 # e.g. "us-gaap:CommonStockSharesOutstanding"
    taxonomy: str                # "us-gaap" | "dei" | "<company-prefix>" for extension tags
    is_extension_tag: bool       # True if taxonomy isn't a standard SEC-published taxonomy
    value: float
    unit: str                    # "USD" | "shares" | "USD-per-shares" | ...
    period_end: str
    period_start: Optional[str]
    form: str                    # filing form type this fact was reported in
    accession_number: str        # dashed, as XBRL API reports it — links back to a documents row
    filed_date: str
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None  # "FY" | "Q1" | "Q2" | "Q3" | "Q4"


@dataclass
class XbrlFieldResult:
    field_key: str
    selected: Optional[XbrlCandidateFact]
    rejected_candidates: List[XbrlCandidateFact] = dc_field(default_factory=list)
    status: str = "not_found"     # "extracted_high_confidence" | "extracted_uncertain" | "not_found" | "conflicting_sources"
    selection_reason: str = ""
    conflict_note: Optional[str] = None  # populated when multiple candidates disagree on value for the same as-of date


class XbrlServiceError(Exception):
    pass


def fetch_company_facts(cik: str) -> dict:
    """Raw SEC XBRL companyfacts JSON for a CIK. Not cached to disk (Phase 4
    will introduce filing-content caching generally; this call is cheap and
    infrequent relative to filing-text downloads)."""
    cik_padded = str(cik).zfill(10)
    resp = _paced_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json")
    return resp.json()


# Candidate concept tags per field, in preference order WITHIN a taxonomy —
# but preference order is only a tie-breaker, never a silent override. Every
# concept present in the company's facts is evaluated as a candidate; this
# list only decides which one gets `selected` when multiple exist and none
# of the disambiguating rules below apply.
#
# One field can pull from multiple taxonomies: "dei" concepts are cover-page
# facts (as-of a specific date, always report_date-driven); "us-gaap"
# concepts are statement facts (period-end driven, GAAP-defined, but with
# company-specific concept choices — hence multiple candidates per field).
_FIELD_TO_XBRL_CONCEPTS: Dict[str, List[str]] = {
    "parent_shares_outstanding": [
        "dei:EntityCommonStockSharesOutstanding",
        "us-gaap:CommonStockSharesOutstanding",
    ],
    "spinoff_shares_outstanding": [
        "dei:EntityCommonStockSharesOutstanding",
        "us-gaap:CommonStockSharesOutstanding",
    ],
    "spinoff_debt": [
        "us-gaap:LongTermDebt",
        "us-gaap:LongTermDebtNoncurrent",
        "us-gaap:LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
        "us-gaap:LongTermDebtAndCapitalLeaseObligations",
        "us-gaap:DebtInstrumentCarryingAmount",
    ],
    "last_year_sales": [
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:Revenues",
        "us-gaap:SalesRevenueNet",
    ],
}

# Fields whose concept is a duration-type fact (reported over a period, not
# as-of an instant) AND whose meaning specifically requires a full fiscal
# year — when multiple duration candidates exist (quarterly, YTD, annual,
# all possibly sharing a period_end), these fields should prefer an
# annual-length one over a shorter one that happens to be more "recent" by
# accession date.
_ANNUAL_FIELDS = {"last_year_sales"}


def _parse_concept(concept: str) -> tuple:
    """'us-gaap:LongTermDebt' -> ('us-gaap', 'LongTermDebt')"""
    if ":" in concept:
        taxonomy, tag = concept.split(":", 1)
        return taxonomy, tag
    return "us-gaap", concept  # bare tag defaults to us-gaap (the common case)


_STANDARD_TAXONOMIES = {"us-gaap", "dei", "srt", "ifrs-full"}


def get_candidates(
    company_facts: dict,
    field_key: str,
    period_end: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    forms: Optional[List[str]] = None,
) -> List[XbrlCandidateFact]:
    """
    Every plausible candidate fact for a field across ALL configured concept
    tags — not just the first one found. Filtering by period_end/fiscal_year/
    forms narrows candidates but never collapses multiple still-matching
    concepts down to one; that collapse only happens in resolve_field().
    """
    concepts = _FIELD_TO_XBRL_CONCEPTS.get(field_key)
    if not concepts:
        raise XbrlServiceError(f"No XBRL concept mapping configured for field_key '{field_key}'")

    candidates: List[XbrlCandidateFact] = []
    facts = company_facts.get("facts", {})

    for concept in concepts:
        taxonomy, tag = _parse_concept(concept)
        taxonomy_facts = facts.get(taxonomy, {})
        if tag not in taxonomy_facts:
            continue
        concept_data = taxonomy_facts[tag]
        units = concept_data.get("units", {})
        for unit, entries in units.items():
            for entry in entries:
                if forms and entry.get("form") not in forms:
                    continue
                if period_end and entry.get("end") != period_end:
                    continue
                if fiscal_year and entry.get("fy") != fiscal_year:
                    continue
                if entry.get("end") is None or entry.get("val") is None:
                    continue
                candidates.append(XbrlCandidateFact(
                    concept=f"{taxonomy}:{tag}",
                    taxonomy=taxonomy,
                    is_extension_tag=taxonomy not in _STANDARD_TAXONOMIES,
                    value=entry["val"],
                    unit=unit,
                    period_end=entry["end"],
                    period_start=entry.get("start"),
                    form=entry.get("form", ""),
                    accession_number=entry.get("accn", ""),
                    filed_date=entry.get("filed", ""),
                    fiscal_year=entry.get("fy"),
                    fiscal_period=entry.get("fp"),
                ))
    return candidates


def resolve_field(
    company_facts: dict,
    field_key: str,
    period_end: Optional[str] = None,
    fiscal_year: Optional[int] = None,
    forms: Optional[List[str]] = None,
) -> XbrlFieldResult:
    """
    Apply deterministic selection over get_candidates()'s full candidate
    list. Selection rule, in order:
      1. No candidates at all -> not_found.
      2. Exactly one candidate -> extracted_high_confidence (nothing to
         disambiguate).
      3. Multiple candidates, all reporting the SAME value for the SAME
         period_end -> extracted_high_confidence, prefer the first concept
         in _FIELD_TO_XBRL_CONCEPTS's preference order as `selected` (a tie-
         break for which accession number to cite, not a claim that the
         others are wrong — they're kept in rejected_candidates).
      4. Multiple candidates with DIFFERING values for the same period_end
         -> conflicting_sources. Nothing is auto-selected; a human must
         resolve which concept's definition actually matches what the
         field means for this filer.
      5. Multiple candidates whose period_end values differ from each other
         (e.g. a dei cover-page date vs. a us-gaap balance-sheet date that
         don't happen to coincide) -> extracted_uncertain, selects the
         candidate closest to the requested period_end if one was given,
         OTHERWISE THE MOST RECENTLY FILED VALUE — which for a company
         filing years after its spin-off is today's figure, not the
         figure as of the spin-off. Confirmed live: called with no
         period_end, this returned GE Vernova's shares outstanding as of
         2026-06-30 (266.3M) when the actual distribution-date figure
         (2024-04-02) was ~274.1M. Field extraction for a specific
         transaction MUST call resolve_field_for_transaction() below, not
         this function directly with period_end=None.
    """
    candidates = get_candidates(company_facts, field_key, period_end, fiscal_year, forms)
    return _select_from_candidates(field_key, candidates, period_end)


def _select_from_candidates(
    field_key: str,
    candidates: List[XbrlCandidateFact],
    period_end: Optional[str] = None,
) -> XbrlFieldResult:
    """Selection logic shared by resolve_field() and
    resolve_field_for_transaction() — operates on an already-filtered
    candidate list rather than raw company_facts, so a caller (like the
    tolerance-window anchor logic below) can filter candidates its own way
    before handing them to the same deterministic selection rule."""
    concept_preference = _FIELD_TO_XBRL_CONCEPTS.get(field_key, [])

    if not candidates:
        return XbrlFieldResult(
            field_key=field_key, selected=None, status="not_found",
            selection_reason="No candidate XBRL facts found for the configured concept tags "
                              "within the given period/form filters.",
        )

    if len(candidates) == 1:
        return XbrlFieldResult(
            field_key=field_key, selected=candidates[0], status="extracted_high_confidence",
            selection_reason="Exactly one candidate fact matched — no other concept tag reported this field.",
        )

    # Group by (period_start, period_end), not period_end alone. A
    # duration-type concept (e.g. revenue) commonly reports BOTH a
    # single-quarter figure and a year-to-date figure sharing the same
    # period_end but different period_start — those are not competing
    # values for the same period, they're different periods that happen to
    # end on the same date. Confirmed live: Grail's
    # RevenueFromContractWithCustomerExcludingAssessedTax reports $31.97M
    # (period_start=2024-04-01, one quarter) and $58.69M
    # (period_start=2024-01-01, six months) both ending 2024-06-30 —
    # comparing period_end alone flagged this as a false conflict.
    def _period_key(c: XbrlCandidateFact) -> tuple:
        return (c.period_start, c.period_end)

    distinct_periods = {_period_key(c) for c in candidates}
    if len(distinct_periods) == 1:
        distinct_values = {c.value for c in candidates}
        if len(distinct_values) == 1:
            selected = min(
                candidates,
                key=lambda c: concept_preference.index(c.concept) if c.concept in concept_preference else 999,
            )
            rejected = [c for c in candidates if c is not selected]
            return XbrlFieldResult(
                field_key=field_key, selected=selected, rejected_candidates=rejected,
                status="extracted_high_confidence",
                selection_reason=f"{len(candidates)} candidate concept tags all report the same value "
                                  f"for period {selected.period_start or ''}..{selected.period_end}; "
                                  f"selected '{selected.concept}' by configured preference order.",
            )
        else:
            return XbrlFieldResult(
                field_key=field_key, selected=None, rejected_candidates=candidates,
                status="conflicting_sources",
                selection_reason=f"{len(candidates)} candidate concept tags report DIFFERENT values "
                                  f"for the same period {next(iter(distinct_periods))} — "
                                  f"requires human review to determine which concept's definition applies.",
                conflict_note="; ".join(f"{c.concept}={c.value}" for c in candidates),
            )

    # Multiple distinct (period_start, period_end) groups exist and none is
    # a clean match. For annual-duration fields this shouldn't normally be
    # reached — resolve_field_for_transaction()'s _resolve_annual_field()
    # already filters to annual-length candidates before calling here — but
    # resolve_field() can still be called directly without that filtering,
    # so fall through to closest-to-anchor as a general-purpose tiebreak.
    #
    # Candidates disagree on period_end itself (e.g. dei as-of-date vs.
    # us-gaap balance-sheet-date facts that don't coincide) — not a value
    # conflict, but not unambiguous either.
    if period_end:
        selected = min(candidates, key=lambda c: abs(
            (int(c.period_end.replace("-", "")) - int(period_end.replace("-", "")))
        ))
        reason = f"Candidates report differing period_end dates; selected the one closest to requested period_end={period_end}."
    else:
        selected = max(candidates, key=lambda c: c.filed_date)
        reason = "Candidates report differing period_end dates and no target period_end was given; selected the most recently filed."
    rejected = [c for c in candidates if c is not selected]
    return XbrlFieldResult(
        field_key=field_key, selected=selected, rejected_candidates=rejected,
        status="extracted_uncertain", selection_reason=reason,
    )


_DEFAULT_SNAPSHOT_TOLERANCE_DAYS = 45   # a quarterly-reporting company's nearest period_end
                                         # to any given date is at most ~45 days away by construction
_ANNUAL_LOOKBACK_TOLERANCE_DAYS = 455   # ~15 months — an annual filing found "before announcement"
                                         # can legitimately be up to a year old (see sec_service.py's
                                         # parent_10k_lookback, same reasoning)
_ANNUAL_DURATION_RANGE_DAYS = (330, 400)  # what counts as "a full fiscal year" duration


def resolve_field_for_transaction(
    company_facts: dict,
    field_key: str,
    distribution_date: Optional[str] = None,
    announcement_date: Optional[str] = None,
    tolerance_days: Optional[int] = None,
    forms: Optional[List[str]] = None,
) -> XbrlFieldResult:
    """
    The safe entry point for extracting an XBRL field FOR A SPECIFIC
    TRANSACTION, instead of drifting to "whatever XBRL most recently
    reported" — which for an actively-filing company is today's figure, not
    the figure as of the spin-off (see resolve_field's docstring for the
    confirmed live example: GEV shares outstanding today vs. at distribution).

    Two different anchoring strategies, chosen by field_key:

    - Snapshot fields (shares outstanding, debt — instant or near-instant
      facts): anchored to distribution_date with a tight
      _DEFAULT_SNAPSHOT_TOLERANCE_DAYS window. These fields mean "as of the
      spin," so a nearby quarterly filing is the right source and a >45-day
      gap should be a visible not_found, not a stale substitution.

    - Annual-duration fields (last_year_sales — "the fiscal year before the
      spin"): anchored to announcement_date instead, searching specifically
      for a ~365-day-duration candidate with period_end at or before
      announcement_date, within a much wider _ANNUAL_LOOKBACK_TOLERANCE_DAYS
      window (~15 months) — annual filings are naturally spaced out, and a
      45-day window around the distribution date will almost always catch
      a quarterly figure instead of the prior fiscal year. Confirmed live:
      before this split existed, last_year_sales for GEV resolved to a
      single Q1 quarter ($7.26B) rather than the prior fiscal year.

    If nothing suitable falls within the relevant tolerance, returns
    not_found rather than silently substituting a stale or wrong-duration
    value.
    """
    if field_key in _ANNUAL_FIELDS:
        if not announcement_date:
            raise XbrlServiceError(
                f"'{field_key}' is an annual-duration field and requires announcement_date, not distribution_date."
            )
        return _resolve_annual_field(company_facts, field_key, announcement_date, forms)

    if not distribution_date:
        raise XbrlServiceError(f"'{field_key}' is a snapshot field and requires distribution_date.")
    return _resolve_snapshot_field(
        company_facts, field_key, distribution_date,
        tolerance_days or _DEFAULT_SNAPSHOT_TOLERANCE_DAYS, forms,
    )


def _resolve_snapshot_field(
    company_facts: dict, field_key: str, anchor_date: str, tolerance_days: int,
    forms: Optional[List[str]],
) -> XbrlFieldResult:
    from datetime import datetime

    all_candidates = get_candidates(company_facts, field_key, forms=forms)
    if not all_candidates:
        return XbrlFieldResult(
            field_key=field_key, selected=None, status="not_found",
            selection_reason="No candidate XBRL facts found for the configured concept tags at all "
                              "(not even outside the anchor tolerance window).",
        )

    anchor_dt = datetime.strptime(anchor_date, "%Y-%m-%d")

    def _days_from_anchor(c: XbrlCandidateFact) -> int:
        return abs((datetime.strptime(c.period_end, "%Y-%m-%d") - anchor_dt).days)

    in_tolerance = [c for c in all_candidates if _days_from_anchor(c) <= tolerance_days]
    if not in_tolerance:
        nearest = min(all_candidates, key=_days_from_anchor)
        return XbrlFieldResult(
            field_key=field_key, selected=None, status="not_found",
            selection_reason=(
                f"No XBRL candidate falls within {tolerance_days} days of anchor_date={anchor_date}. "
                f"Nearest available candidate ({nearest.concept}, period_end={nearest.period_end}) is "
                f"{_days_from_anchor(nearest)} days away — too stale to auto-select; flag for manual "
                f"sourcing from the Form 10 capitalization table instead."
            ),
        )

    # Prefer a nonzero candidate over the literal nearest-by-days one. A
    # snapshot fact of exactly 0 right before a spin-off distribution is
    # near-certainly a pre-existence artifact (the entity has no publicly
    # issued shares/balances yet), not a real "0" — confirmed live: Grail's
    # spinoff_shares_outstanding picked a period_end 2 days BEFORE
    # distribution reporting 0 shares over one 5 days AFTER reporting the
    # real post-distribution count (31,049,148), purely because 2 < 5 days.
    # Falls back to the plain nearest-by-days pick (which may be 0) only if
    # every in-tolerance candidate is 0 — a field that's genuinely zero
    # throughout the window shouldn't be masked by this preference.
    nonzero_in_tolerance = [c for c in in_tolerance if c.value != 0]
    ranked = nonzero_in_tolerance or in_tolerance
    nearest_in_tolerance = min(ranked, key=_days_from_anchor)
    return _select_from_candidates(field_key, in_tolerance, period_end=nearest_in_tolerance.period_end)


def _resolve_annual_field(
    company_facts: dict, field_key: str, announcement_date: str, forms: Optional[List[str]],
) -> XbrlFieldResult:
    from datetime import datetime

    all_candidates = get_candidates(company_facts, field_key, forms=forms)
    if not all_candidates:
        return XbrlFieldResult(
            field_key=field_key, selected=None, status="not_found",
            selection_reason="No candidate XBRL facts found for the configured concept tags at all.",
        )

    def _duration_days(c: XbrlCandidateFact) -> Optional[int]:
        if not c.period_start:
            return None
        return (datetime.strptime(c.period_end, "%Y-%m-%d") - datetime.strptime(c.period_start, "%Y-%m-%d")).days

    lo, hi = _ANNUAL_DURATION_RANGE_DAYS
    annual_candidates = [c for c in all_candidates if (d := _duration_days(c)) is not None and lo <= d <= hi]
    if not annual_candidates:
        return XbrlFieldResult(
            field_key=field_key, selected=None, status="not_found",
            selection_reason=f"No candidate fact has an annual-length duration ({lo}-{hi} days) — "
                              f"only quarterly/YTD figures are available for this concept.",
        )

    announcement_dt = datetime.strptime(announcement_date, "%Y-%m-%d")

    def _days_before_announcement(c: XbrlCandidateFact) -> Optional[int]:
        end_dt = datetime.strptime(c.period_end, "%Y-%m-%d")
        delta = (announcement_dt - end_dt).days
        return delta if delta >= 0 else None  # exclude periods ending AFTER announcement — not "last year"

    eligible = [c for c in annual_candidates if _days_before_announcement(c) is not None]
    in_tolerance = [c for c in eligible if _days_before_announcement(c) <= _ANNUAL_LOOKBACK_TOLERANCE_DAYS]
    if not in_tolerance:
        return XbrlFieldResult(
            field_key=field_key, selected=None, status="not_found",
            selection_reason=(
                f"No annual-duration candidate ends within {_ANNUAL_LOOKBACK_TOLERANCE_DAYS} days before "
                f"announcement_date={announcement_date}."
            ),
        )

    nearest = min(in_tolerance, key=_days_before_announcement)
    return _select_from_candidates(field_key, in_tolerance, period_end=nearest.period_end)

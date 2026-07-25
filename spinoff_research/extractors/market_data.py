"""
yFinance-backed extractors: parent/spinoff sector, industry, share_price.

Confidence and convention choices here are NOT arbitrary — they follow
directly from live validation against Rich's own 3 pilot values:
  - sector/industry: yfinance returns real, correct classifications, but
    its taxonomy doesn't always match Rich's own labels (confirmed: his
    "Life Sciences Tools & Diagnostics" vs yfinance's "Diagnostics &
    Research" for ILMN — same company, different vocabulary). Always
    extracted_uncertain, never high-confidence, even though the mechanism
    itself works.
  - share_price: checked all 3 pilots against Rich's recorded values.
    GE matched the day BEFORE distribution; ILMN matched the day OF
    distribution exactly; SNY's recorded value matched no close price in
    5 years of history under any date/venue tried (likely a data-entry
    error on his end, not a convention to reproduce). No single date rule
    explains the confirmed cases, so this extractor uses distribution-date
    close (the more defensible standalone convention — "price as of the
    spin") and ALWAYS tags the result extracted_uncertain. This field
    needs Rich's review every time, not blind trust either way.
"""
from typing import List, Optional

from spinoff_research.extraction import ExtractedFieldValue, SourceCitation
from spinoff_research.status import FieldStatus


def extract_sector_industry(ticker: str, field_key_prefix: str) -> List[ExtractedFieldValue]:
    """Returns [sector_result, industry_result] for 'parent' or 'spinoff'."""
    sector_key = f"{field_key_prefix}_sector"
    industry_key = f"{field_key_prefix}_industry"

    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        sector = info.get("sector")
        industry = info.get("industry")
    except Exception as e:
        not_found = FieldStatus.NOT_FOUND
        note = SourceCitation(reasoning_summary=f"yfinance lookup failed for '{ticker}': {e}")
        return [
            ExtractedFieldValue(field_key=sector_key, extraction_method="market_data", status=not_found, sources=[note]),
            ExtractedFieldValue(field_key=industry_key, extraction_method="market_data", status=not_found, sources=[note]),
        ]

    citation = SourceCitation(
        reasoning_summary=(
            f"yfinance sector/industry classification for '{ticker}'. Uses Yahoo Finance's own "
            f"taxonomy, which may not match Rich's classification labels exactly even when correct."
        ),
    )

    results = []
    for key, value in [(sector_key, sector), (industry_key, industry)]:
        if value:
            results.append(ExtractedFieldValue(
                field_key=key, extraction_method="market_data",
                status=FieldStatus.EXTRACTED_UNCERTAIN,  # always — see module docstring
                raw_value=value, normalized_value=value,
                confidence=0.6, sources=[citation],
            ))
        else:
            results.append(ExtractedFieldValue(field_key=key, extraction_method="market_data", status=FieldStatus.NOT_FOUND))
    return results


def extract_share_price(ticker: str, field_key: str, distribution_date: str) -> ExtractedFieldValue:
    """
    field_key: 'parent_share_price' or 'spinoff_share_price'.
    Pulls the UNADJUSTED close on distribution_date (see module docstring
    for why this convention, not day-before, and why confidence is capped
    regardless of whether a value is found).
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        from datetime import datetime, timedelta
        start = distribution_date
        end = (datetime.strptime(distribution_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        hist = t.history(start=start, end=end, auto_adjust=False)
    except Exception as e:
        return ExtractedFieldValue(
            field_key=field_key, extraction_method="market_data", status=FieldStatus.NOT_FOUND,
            sources=[SourceCitation(reasoning_summary=f"yfinance lookup failed for '{ticker}': {e}")],
        )

    if hist.empty:
        return ExtractedFieldValue(
            field_key=field_key, extraction_method="market_data", status=FieldStatus.NOT_FOUND,
            sources=[SourceCitation(
                reasoning_summary=f"No trading data for '{ticker}' on {distribution_date} "
                                  f"(non-trading day, or ticker not yet listed)."
            )],
        )

    close = float(hist["Close"].iloc[0])
    citation = SourceCitation(
        supporting_excerpt=f"{ticker} close on {distribution_date}: ${close:.2f}",
        reasoning_summary=(
            "yfinance unadjusted close on the distribution date. Convention chosen because it matched "
            "one confirmed pilot exactly (ILMN); another pilot (GE) matched the PRIOR trading day instead "
            "and a third (SNY) matched no date under any convention tried — Rich's own values are not "
            "methodologically consistent for this field, so this is always extracted_uncertain."
        ),
    )
    return ExtractedFieldValue(
        field_key=field_key, extraction_method="market_data",
        status=FieldStatus.EXTRACTED_UNCERTAIN,
        raw_value=f"{close:.2f}", numeric_value=close, unit="USD", currency="USD",
        as_of_date=distribution_date, confidence=0.5,
        sources=[citation],
    )

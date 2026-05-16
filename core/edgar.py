import re
import requests
import streamlit as st
from typing import Dict, Tuple, Optional


def _strip_html(html: str) -> str:
    """Strip HTML/iXBRL tags, skip script/style blocks, preserve whitespace at block boundaries."""
    from html.parser import HTMLParser

    # Block-level tags that start a new line; closing these also adds a newline
    _BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5",
              "h6", "section", "article", "header", "footer", "main", "aside",
              "table", "thead", "tbody", "tfoot", "blockquote"}

    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self._parts: list = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "head"):
                self._skip = True
            elif tag in _BLOCK:
                self._parts.append("\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style", "head"):
                self._skip = False
            elif tag in _BLOCK:
                self._parts.append("\n")

        def handle_data(self, data):
            if not self._skip:
                self._parts.append(data.replace("\xa0", " "))

    stripper = _Stripper()
    stripper.feed(html)
    text = "".join(stripper._parts)
    # Collapse 3+ newlines to 2 (paragraph break), then collapse lone \n with space
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)   # single \n → space (inline breaks)
    return re.sub(r' {2,}', ' ', text).strip()


def fetch_edgar_filing(ticker: str, form_type: str = "10-K",
                       target_year: str = None) -> Tuple[bool, str, str, str, str]:
    headers = {"User-Agent": "MeridianPlatform student@meridian.edu"}
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=10)
        tickers_data = r.json()
        cik, company_name = None, None
        for entry in tickers_data.values():
            if entry["ticker"].upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                company_name = entry["title"]
                break
        if not cik:
            return False, "", f"Ticker '{ticker}' not found in SEC EDGAR", "", ""

        r2 = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers, timeout=10)
        sub = r2.json()
        filings = sub.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        accessions = filings.get("accessionNumber", [])
        dates = filings.get("filingDate", [])
        report_dates = filings.get("reportDate", [])  # fiscal period end date
        primary_docs = filings.get("primaryDocument", [])  # e.g. "goog-20231231.htm"

        if target_year:
            # Match by reportDate (fiscal year end) — e.g. target_year="2023" matches "2023-12-31"
            idx = next(
                (i for i, (f, rd) in enumerate(zip(forms, report_dates))
                 if f == form_type and str(rd).startswith(target_year)),
                None
            )
            if idx is None:
                # Surface the available years so the user knows what to ask for
                available = sorted(
                    {str(rd)[:4] for f, rd in zip(forms, report_dates) if f == form_type and rd},
                    reverse=True
                )
                avail_str = ", ".join(available) if available else "none found"
                return False, "", (
                    f"No {form_type} found for {ticker} with fiscal year {target_year}. "
                    f"Available years: {avail_str}"
                ), "", ""
        else:
            idx = next((i for i, f in enumerate(forms) if f == form_type), None)
            if idx is None:
                return False, "", f"No {form_type} found for {ticker}", "", ""

        if idx >= len(accessions) or idx >= len(dates):
            return False, "", f"SEC data for {ticker} is inconsistent (lists have different lengths). Try a different form type.", "", ""

        raw_acc = accessions[idx]
        acc_no = raw_acc.replace("-", "")
        filing_date = dates[idx]
        # HTML is clean stripped text — 3M chars covers the largest 10-Ks in full.
        # The .txt SGML bundle contains boilerplate noise so a lower cap is fine.
        HTML_CAP = 3_000_000
        TXT_CAP  =   600_000

        # Try primary HTML document first — cleaner than the raw .txt SGML bundle
        clean = None
        char_cap = HTML_CAP
        primary_doc = primary_docs[idx] if idx < len(primary_docs) else ""
        if primary_doc and primary_doc.lower().endswith((".htm", ".html")):
            html_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/{primary_doc}"
            try:
                rh = requests.get(html_url, headers=headers, timeout=30)
                if rh.status_code == 200:
                    clean = _strip_html(rh.text)
                    st.caption(f"📄 Fetched primary HTML document: `{primary_doc}`")
            except Exception:
                pass  # fall through to .txt

        # Fallback: raw .txt SGML submission bundle
        if not clean:
            char_cap = TXT_CAP
            text_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/{raw_acc}.txt"
            r3 = requests.get(text_url, headers=headers, timeout=30)
            if r3.status_code != 200:
                return False, "", f"Download failed (HTTP {r3.status_code}). Upload the PDF manually.", "", ""
            clean = re.sub(r'<[^>]+>', ' ', r3.text)
            st.caption("📄 Fetched raw .txt submission bundle (HTML fallback unavailable)")

        if len(clean) > char_cap:
            st.warning(f"Document truncated to {char_cap:,} chars for processing. Full filing may contain more.")
        clean = clean[:char_cap]
        return True, clean, f"{company_name} {form_type} ({filing_date})", cik, company_name
    except Exception as e:
        return False, "", f"EDGAR error: {e}", "", ""


def fetch_xbrl_facts(ticker: str) -> Tuple[bool, Dict, str]:
    """
    Fetch structured financial facts from SEC XBRL Company Facts API.
    Returns (success, {metric_label: [entries]}, company_name)
    Each entry: {value, period_end, period_start, form, filed, period}
    """
    headers = {"User-Agent": "MeridianPlatform student@meridian.edu"}
    # Priority-ordered list of GAAP concept names for each human-readable metric.
    # The first concept found in the company's XBRL data is used.
    CONCEPT_MAP = {
        "Revenue":             ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
        "Net Income":          ["NetIncomeLoss"],
        "Operating Income":    ["OperatingIncomeLoss"],
        "Gross Profit":        ["GrossProfit"],
        "Total Assets":        ["Assets"],
        "Total Liabilities":   ["Liabilities"],
        "Stockholders Equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        "Operating Cash Flow": ["NetCashProvidedByUsedInOperatingActivities"],
        "Cash & Equivalents":  ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsAndShortTermInvestments"],
        "EPS (Diluted)":       ["EarningsPerShareDiluted"],
        "Long-Term Debt":      ["LongTermDebt", "LongTermDebtNoncurrent"],
        "R&D Expense":         ["ResearchAndDevelopmentExpense"],
    }
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=10)
        r.raise_for_status()
        cik, company_name = None, None
        for entry in r.json().values():
            if entry["ticker"].upper() == ticker.upper():
                cik = str(entry["cik_str"]).zfill(10)
                company_name = entry["title"]
                break
        if not cik:
            return False, {}, f"Ticker '{ticker}' not found in SEC EDGAR"

        r2 = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers=headers, timeout=20
        )
        r2.raise_for_status()
        us_gaap = r2.json().get("facts", {}).get("us-gaap", {})

        facts = {}
        for label, concepts in CONCEPT_MAP.items():
            for concept in concepts:
                if concept not in us_gaap:
                    continue
                units = us_gaap[concept].get("units", {})
                raw = units.get("USD") or units.get("shares") or []
                entries = [
                    {
                        "value":        v["val"],
                        "period_end":   v.get("end", ""),
                        "period_start": v.get("start", ""),
                        "form":         v.get("form", ""),
                        "filed":        v.get("filed", ""),
                        "period":       v.get("fp", ""),
                    }
                    for v in raw
                    if v.get("form") in ("10-K", "10-Q") and v.get("end")
                ]
                if entries:
                    entries.sort(key=lambda x: x["period_end"], reverse=True)
                    facts[label] = entries[:16]  # ~4 years of data
                    break
        return True, facts, company_name
    except Exception as e:
        return False, {}, f"XBRL error: {e}"


def _fmt_xbrl(value: float, label: str) -> str:
    """Scale raw XBRL USD/share values to readable strings."""
    if "EPS" in label:
        return f"${value:.2f}"
    if abs(value) >= 1e12:
        return f"${value / 1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.1f}M"
    return f"${value:,.0f}"


def _detect_url_metadata(url: str, html_text: str) -> dict:
    """
    Auto-detect ticker, form_type, fiscal_year, company, cik from a filing URL + its HTML.
    For SEC EDGAR URLs the submissions API gives exact values.
    For other URLs we scan the first 5,000 chars of HTML content.
    """
    from core.chunking import _parse_filename_metadata

    meta = {"ticker": "", "form_type": "", "fiscal_year": "", "company": "", "cik": ""}
    hdrs = {"User-Agent": "MeridianPlatform student@meridian.edu"}

    # ── EDGAR URL: /Archives/edgar/data/{cik}/{18-digit-accno}/ ──────────────
    edgar_match = re.search(r'/Archives/edgar/data/(\d+)/(\d{18})/', url)
    if edgar_match:
        cik_int   = edgar_match.group(1)
        acc_nodash = edgar_match.group(2)
        cik       = str(cik_int).zfill(10)
        raw_acc   = f"{acc_nodash[0:10]}-{acc_nodash[10:12]}-{acc_nodash[12:]}"
        meta["cik"] = cik
        try:
            r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=hdrs, timeout=10)
            sub = r.json()
            meta["company"] = sub.get("name", "")
            tickers = sub.get("tickers", [])
            meta["ticker"] = tickers[0].upper() if tickers else ""
            filings = sub.get("filings", {}).get("recent", {})
            accessions = filings.get("accessionNumber", [])
            if raw_acc in accessions:
                idx = accessions.index(raw_acc)
                forms        = filings.get("form", [])
                report_dates = filings.get("reportDate", [])
                if idx < len(forms):
                    meta["form_type"] = forms[idx]
                if idx < len(report_dates) and report_dates[idx]:
                    meta["fiscal_year"] = str(report_dates[idx])[:4]
        except Exception:
            pass  # fall through to HTML analysis

    # ── HTML content: fill gaps using cover-page text ─────────────────────
    cover = html_text[:5000]
    if not meta["form_type"]:
        cl = cover.lower()
        if "annual report" in cl or "form 10-k" in cl:
            meta["form_type"] = "10-K"
        elif "quarterly report" in cl or "form 10-q" in cl:
            meta["form_type"] = "10-Q"
        elif "registration statement" in cl or "form 10 " in cl:
            meta["form_type"] = "Form 10"
        else:
            meta["form_type"] = "10-K"

    if not meta["fiscal_year"]:
        m = re.search(
            r'(?:fiscal year|year|period|quarter)\s+ended?\s+\w+\.?\s+\d{1,2},?\s+(\d{4})',
            cover, re.IGNORECASE
        )
        if m:
            meta["fiscal_year"] = m.group(1)

    # ── Filename fallback ─────────────────────────────────────────────────
    filename = url.rstrip("/").split("/")[-1]
    fname_meta = _parse_filename_metadata(filename)
    if not meta["ticker"]      and fname_meta.get("ticker"):      meta["ticker"]      = fname_meta["ticker"]
    if not meta["fiscal_year"] and fname_meta.get("fiscal_year"): meta["fiscal_year"] = fname_meta["fiscal_year"]
    if not meta["form_type"]   and fname_meta.get("form_type"):   meta["form_type"]   = fname_meta["form_type"]

    return meta

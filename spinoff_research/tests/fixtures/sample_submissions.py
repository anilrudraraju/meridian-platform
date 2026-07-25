"""
Synthetic SEC submissions-API and company_tickers.json fixtures for testing
sec_service.py without live network calls. Shapes match the real API
responses observed against GE Vernova (CIK 0001996810), Sanofi/Inhibrx
(FPI parent case), and Illumina/Grail during Phase 3 development — trimmed
to a handful of representative filings per form type rather than full
multi-hundred-entry histories.
"""

SAMPLE_TICKER_MAP_RAW = {
    "0": {"cik_str": 40545, "ticker": "GE", "title": "GENERAL ELECTRIC CO"},
    "1": {"cik_str": 1996810, "ticker": "GEV", "title": "GE Vernova Inc."},
    "2": {"cik_str": 1121404, "ticker": "SNY", "title": "Sanofi"},
    "3": {"cik_str": 2007919, "ticker": "INBX", "title": "Inhibrx Biosciences, Inc."},
}

# Domestic filer (GE Vernova) — has 10-12B, 10-K, 10-Q, DEF 14A, 8-K, Form 4
SAMPLE_SUBMISSIONS_DOMESTIC = {
    "name": "GE Vernova Inc.",
    "sic": "3600",
    "filings": {
        "recent": {
            "form":            ["10-12B/A",      "10-12B",        "10-Q",          "DEF 14A",       "10-K",          "8-K",           "4",             "4",             "SC 13G"],
            "filingDate":      ["2024-03-05",     "2024-02-15",    "2024-08-01",    "2025-04-03",    "2025-02-03",    "2024-05-01",    "2024-04-04",    "2025-06-01",    "2024-04-10"],
            "reportDate":      ["",               "",              "2024-06-30",    "",              "2024-12-31",    "",              "",              "",              ""],
            "accessionNumber": ["0001193125-24-059354", "0001193125-24-037526", "0001996810-24-000050", "0001996810-25-000049", "0001996810-25-000010", "0001996810-24-000030", "0001996810-24-000038", "0001996810-25-000037", "0001234567-24-000001"],
            # Form 4 primaryDocument paths are 'xslF345X02/<file>.xml' in the
            # real API response — that's an SEC XSL-rendered HTML VIEW path,
            # not the raw file location, confirmed live across all 3 pilots.
            "primaryDocument": ["d542465d1012ba.htm", "d542465d1012b.htm", "gev-20240630.htm", "gev-20250402.htm", "gev-20241231.htm", "gev-20240501.htm", "xslF345X02/wk-form4.xml", "xslF345X02/wk-form4.xml", "sc13g.htm"],
        }
    },
}

# Foreign private issuer parent (Sanofi) — 20-F/6-K instead of 10-K/8-K.
# 20-F filing date is deliberately BEFORE the test's announcement_date
# (2024-01-23) — a "most recent annual report before announcement" is
# realistically filed weeks-to-months earlier, same as GE's real 10-K
# (filed 2021-02-12) predating the GEV announcement (2021-11-09) by 9 months.
SAMPLE_SUBMISSIONS_FPI = {
    "name": "Sanofi",
    "sic": "2836",
    "filings": {
        "recent": {
            "form":            ["20-F",            "6-K",           "6-K",           "4"],
            "filingDate":      ["2023-02-01",       "2024-01-24",    "2024-06-01",    "2024-06-05"],
            "reportDate":      ["2022-12-31",       "",              "",              ""],
            "accessionNumber": ["0001121404-23-000010", "0001121404-24-000015", "0001121404-24-000090", "0001121404-24-000091"],
            "primaryDocument": ["sny-20221231.htm", "sny-6k-0124.htm", "sny-6k-0601.htm", "form4.xml"],
        }
    },
}

# 10-K and Form 10 section patterns — Item number optional, title required
SECTION_PATTERNS_10K = {
    "business":         r"item\s*1[.\s]+business",
    "risk_factors":     r"item\s*1a[.\s]+risk\s*factor",
    "properties":       r"item\s*2[.\s]+propert",
    "legal":            r"item\s*3[.\s]+legal",
    "mdna":             r"item\s*7[.\s]+management",
    "quantitative":     r"item\s*7a[.\s]+quantitative",
    "financial_stmts":  r"item\s*8[.\s]+financial\s*state",
    "controls":         r"item\s*9[.\s]",
    "footnotes":        r"notes\s+to\s+(consolidated\s+)?financial",
}

# Form 10 extra sections not present in 10-K/10-Q
SECTION_PATTERNS_FORM10_EXTRA = {
    "dilution":               r"dilution",
    "use_of_proceeds":        r"use\s+of\s+proceeds",
    "capitalization":         r"capitalization",
    "selected_financials":    r"selected\s+(consolidated\s+)?financial\s+data",
    "related_party":          r"related\s+(party|person)\s+transactions",
    "description_securities": r"description\s+of\s+(capital\s+)?securities",
}

# 10-Q uses Part I / Part II item numbering — different from 10-K
SECTION_PATTERNS_10Q = {
    "financial_stmts": r"(?:part\s*i\s+)?item\s*1[.\s]+financial\s*state",
    "mdna":            r"(?:part\s*i\s+)?item\s*2[.\s]+management",
    "quantitative":    r"(?:part\s*i\s+)?item\s*3[.\s]+quantitative",
    "controls":        r"(?:part\s*i\s+)?item\s*4[.\s]+controls",
    "legal":           r"(?:part\s*ii\s+)?item\s*1[.\s]+legal",
    "risk_factors":    r"(?:part\s*ii\s+)?item\s*1a[.\s]+risk",
    "footnotes":       r"notes\s+to\s+(condensed\s+)?(consolidated\s+)?financial",
}

# Financial statement header patterns — match near line starts (within 80 chars)
STATEMENT_PATTERNS = {
    "income_statement":     r"consolidated\s+statements?\s+of\s+(?:operations|income)",
    "balance_sheet":        r"consolidated\s+(?:balance\s+sheets?|statements?\s+of\s+financial\s+position)",
    "cash_flow":            r"consolidated\s+statements?\s+of\s+cash\s+flows?",
    "equity":               r"statements?\s+of\s+(?:stockholders?[\s\']?\s*equity|changes\s+in\s+equity)",
    "comprehensive_income": r"statements?\s+of\s+comprehensive\s+income",
}

# MD&A sub-section headers
MDNA_SUBSECTION_PATTERNS = [
    ("overview",             r"overview"),
    ("results",              r"results\s+of\s+operations"),
    ("liquidity",            r"liquidity\s+and\s+capital"),
    ("critical_accounting",  r"critical\s+accounting"),
    ("contractual",          r"contractual\s+obligations"),
    ("off_balance",          r"off.balance\s+sheet"),
    ("market_risk",          r"market\s+risk"),
    ("recent_accounting",    r"recently\s+issued\s+accounting"),
]

# Query router keyword sets
STRUCTURED_SIGNALS = {
    "how much", "what was", "what were", "revenue", "income", "earnings",
    "profit", "loss", "assets", "debt", "cash", "margin", "grew", "declined",
    "increased", "decreased", "percent", "quarter", "fiscal year", "compared to",
    "how many", "total", "net", "gross",
}
NARRATIVE_SIGNALS = {
    "why", "how did", "explain", "describe", "what are", "risk", "strategy",
    "outlook", "reason", "factor", "management", "competitive", "concern",
    "discuss", "what caused", "what drove",
}

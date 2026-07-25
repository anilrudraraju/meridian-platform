"""
Normalized field data dictionary for the spin-off research prototype.

Source of truth for field metadata: data/input/rich_spinoff_database.xlsx
("Field Source Review" worksheet, 47 rows, initial hypothesis only —
Rich has not yet returned corrections). This module is the machine-readable
version of that sheet, corrected where the sheet's own "Structured/
Deterministic" guess violates the extraction-category rules below.

Corrections applied vs. the raw worksheet (see class docstring on each
field below for the "why"):
    - spin-off EBITDA margin       : structured -> ai_assisted
    - spin-off debt-to-EBITDA      : structured -> ai_assisted
    - ROIC                          : structured -> ai_assisted
    - dis-synergy % of prior sales : structured -> calculated (depends on
      an AI-assisted input, so the composite cannot be "deterministic")

EBITDA and ROIC are not standardized XBRL/GAAP tags — every issuer
computes them differently (which adjustments, which capital base), so
"pull the XBRL fact" is not a valid strategy for these three fields.
They require reading the company's own definition (usually in an
earnings release or MD&A non-GAAP reconciliation) and are therefore
AI-assisted-with-citation at best, not deterministic.
"""
from dataclasses import dataclass, field as dc_field
from enum import Enum
from typing import List, Optional


class ExtractionCategory(str, Enum):
    STRUCTURED = "structured_or_deterministic"
    AI_ASSISTED = "ai_assisted_with_citation"
    SCHEDULED_MANUAL = "scheduled_or_manual"
    CALCULATED = "calculated"  # composite of other field_values; category matches its weakest input


class SourceType(str, Enum):
    XBRL = "XBRL"
    FILING_METADATA = "FILING_METADATA"
    FORM_10 = "FORM_10"
    INFORMATION_STATEMENT = "INFORMATION_STATEMENT"  # Form 10 Exhibit 99.1
    FORM_8K = "8-K"
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    DEF_14A = "DEF_14A"
    FORM_4 = "FORM_4"
    PRESS_RELEASE = "PRESS_RELEASE"
    INVESTOR_PRESENTATION = "INVESTOR_PRESENTATION"
    MARKET_DATA = "MARKET_DATA"
    EARNINGS_CALL_TRANSCRIPT = "EARNINGS_CALL_TRANSCRIPT"
    INTERNAL = "INTERNAL"  # e.g. sequence number assigned at intake, not extracted from a document


@dataclass
class FieldDefinition:
    field_key: str
    display_name: str
    spreadsheet_column_name: str
    category: str                       # Rich's group, e.g. "transaction_mechanics"
    description: str
    data_type: str                      # "string" | "date" | "ratio" | "currency" | "int" | "bool" | "enum" | "percent"
    extraction_category: ExtractionCategory
    preferred_source_types: List[SourceType]
    alternate_source_types: List[SourceType] = dc_field(default_factory=list)
    allowed_values: Optional[List[str]] = None
    unit: Optional[str] = None
    search_terms: List[str] = dc_field(default_factory=list)
    structured_extraction_strategy: Optional[str] = None
    ai_extraction_instructions: Optional[str] = None
    calculation_formula: Optional[str] = None
    required_inputs: List[str] = dc_field(default_factory=list)   # other field_keys, for calculated fields
    required_observation_window_days: Optional[int] = None         # for scheduled_or_manual fields
    requires_citation: bool = True
    requires_human_review: bool = True
    validation_rules: List[str] = dc_field(default_factory=list)
    notes: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────
# Basic identification
# ─────────────────────────────────────────────────────────────────────────

FIELD_DEFINITIONS: List[FieldDefinition] = [
    FieldDefinition(
        field_key="internal_sequence_number",
        display_name="Number",
        spreadsheet_column_name="Number",
        category="basic_identification",
        description="Internal row/sequence identifier assigned when a transaction is added to the research list. Not extracted from any document.",
        data_type="int",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.INTERNAL],
        requires_citation=False,
        requires_human_review=False,
    ),
    FieldDefinition(
        field_key="resources",
        display_name="Resources",
        spreadsheet_column_name="Resources",
        category="basic_identification",
        description="Free-text notes/links a reviewer records about where they looked. Human-entered, not extracted.",
        data_type="string",
        extraction_category=ExtractionCategory.SCHEDULED_MANUAL,
        preferred_source_types=[SourceType.INTERNAL],
        requires_citation=False,
        requires_human_review=True,
    ),
    FieldDefinition(
        field_key="form_10_availability",
        display_name="Form 10 availability",
        spreadsheet_column_name="Form 10 availability",
        category="basic_identification",
        description="Whether a Form 10 (or Form 10 amendment / information statement) is available on EDGAR for this transaction.",
        data_type="bool",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.FILING_METADATA],
        structured_extraction_strategy="EDGAR full-text search + submissions API filtered to form types 10-12B/10-12G/10-12B/A for the spinco CIK.",
        search_terms=["10-12B", "10-12G", "10-12B/A", "Form 10"],
        requires_citation=True,
        requires_human_review=False,
        notes="Pre-2001 spin-offs may predate EDGAR full-text search; may require manual EDGAR browse fallback.",
    ),
    FieldDefinition(
        field_key="parent_company_name",
        display_name="Parent company",
        spreadsheet_column_name="Parent company",
        category="basic_identification",
        description="Legal name of the parent company at the time of the spin-off.",
        data_type="string",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.FILING_METADATA],
        alternate_source_types=[SourceType.FORM_10],
        structured_extraction_strategy="EDGAR company search / CIK lookup (company_tickers.json or submissions API 'name' field).",
        requires_human_review=False,
    ),
    FieldDefinition(
        field_key="spinoff_company_name",
        display_name="Spun-off company",
        spreadsheet_column_name="Spun-off company",
        category="basic_identification",
        description="Legal name of the spun-off company.",
        data_type="string",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.FILING_METADATA],
        alternate_source_types=[SourceType.FORM_10],
        requires_human_review=False,
    ),
    FieldDefinition(
        field_key="parent_ticker",
        display_name="Parent ticker",
        spreadsheet_column_name="Parent ticker",
        category="basic_identification",
        description="Parent company's exchange ticker symbol at the time of the spin-off.",
        data_type="string",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.FILING_METADATA],
        alternate_source_types=[SourceType.FORM_10],
        requires_human_review=False,
        notes="Ticker may have changed since the spin-off; store as-of-spin-date ticker.",
    ),
    FieldDefinition(
        field_key="spinoff_ticker",
        display_name="Spin-off ticker",
        spreadsheet_column_name="Spin-off ticker",
        category="basic_identification",
        description="Spun-off company's exchange ticker symbol upon listing.",
        data_type="string",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.FILING_METADATA],
        alternate_source_types=[SourceType.FORM_10],
        requires_human_review=False,
    ),

    # ─────────────────────────────────────────────────────────────────
    # Transaction mechanics
    # ─────────────────────────────────────────────────────────────────
    FieldDefinition(
        field_key="announcement_date",
        display_name="Spin-off announcement date",
        spreadsheet_column_name="Spin-off announcement date",
        category="transaction_mechanics",
        description="Date the parent publicly announced intent to spin off the business.",
        data_type="date",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_8K],
        alternate_source_types=[SourceType.PRESS_RELEASE],
        search_terms=["announced", "intends to separate", "plans to spin off", "board approved"],
        ai_extraction_instructions="Find the date the 8-K or press release states the separation was announced (not filed date — the announcement date stated in the text).",
        validation_rules=["announcement_date <= distribution_date (if known)"],
    ),
    FieldDefinition(
        field_key="regular_way_trading_date",
        display_name="Beginning of regular-way trading",
        spreadsheet_column_name="Beginning of regular-way trading",
        category="transaction_mechanics",
        description="First date the spun-off company's shares trade regular-way (T+2 settlement) on an exchange.",
        data_type="date",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_8K],
        alternate_source_types=[SourceType.FORM_10],
        search_terms=["regular-way trading", "when-issued", "distribution date"],
        ai_extraction_instructions="Distinguish 'regular-way trading' date from 'when-issued' trading date — these are often different dates in the same filing.",
    ),
    FieldDefinition(
        field_key="distribution_ratio",
        display_name="Distribution ratio",
        spreadsheet_column_name="Distribution ratio",
        category="transaction_mechanics",
        description="Ratio of spinco shares distributed per parent share held (e.g. 1 spinco share for every 4 parent shares).",
        data_type="ratio",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10, SourceType.INFORMATION_STATEMENT],
        alternate_source_types=[SourceType.FORM_8K],
        search_terms=["for every", "distribution ratio", "shares of common stock", "one share of"],
        ai_extraction_instructions="Extract as a normalized ratio (spinco_shares : parent_shares), e.g. '1:4'. Watch for ratios stated the opposite way round.",
        requires_human_review=True,
    ),
    FieldDefinition(
        field_key="parent_shares_outstanding",
        display_name="Parent shares outstanding",
        spreadsheet_column_name="Parent shares outstanding",
        category="transaction_mechanics",
        description="Parent company shares outstanding as of (or nearest to) the distribution date.",
        data_type="int",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.XBRL],
        alternate_source_types=[SourceType.FORM_10],
        structured_extraction_strategy="XBRL dei:EntityCommonStockSharesOutstanding on the 10-K/10-Q cover page nearest the distribution date.",
        requires_human_review=False,
    ),
    FieldDefinition(
        field_key="parent_share_price",
        display_name="Parent share price",
        spreadsheet_column_name="Parent share price",
        category="transaction_mechanics",
        description="Parent share price on a defined reference date (distribution date close, unless specified otherwise).",
        data_type="currency",
        unit="USD",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.MARKET_DATA],
        structured_extraction_strategy="Market data API (e.g. yfinance, already a Meridian dependency) — close price on distribution date.",
        requires_human_review=False,
        validation_rules=["price_date must be explicit and stored alongside the value"],
    ),
    FieldDefinition(
        field_key="parent_market_cap",
        display_name="Parent market capitalization",
        spreadsheet_column_name="Parent market capitalization",
        category="transaction_mechanics",
        description="Parent market capitalization = parent_share_price x parent_shares_outstanding, as of the same reference date.",
        data_type="currency",
        unit="USD",
        extraction_category=ExtractionCategory.CALCULATED,
        preferred_source_types=[],
        calculation_formula="parent_share_price * parent_shares_outstanding",
        required_inputs=["parent_share_price", "parent_shares_outstanding"],
        requires_human_review=False,
        validation_rules=["both inputs must share the same as-of date, or the mismatch must be flagged"],
    ),
    FieldDefinition(
        field_key="spinoff_shares_outstanding",
        display_name="Spin-off shares outstanding",
        spreadsheet_column_name="Spin-off shares outstanding",
        category="transaction_mechanics",
        description="Spinco shares outstanding immediately after distribution.",
        data_type="int",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.FORM_10],
        alternate_source_types=[SourceType.XBRL, SourceType.FORM_8K],
        structured_extraction_strategy="Form 10 capitalization table, or first 10-K/10-Q cover-page XBRL fact once filed.",
        requires_human_review=True,
        notes="XBRL not available until spinco's first post-spin filing; Form 10 capitalization table is the only source at distribution time.",
    ),
    FieldDefinition(
        field_key="spinoff_share_price",
        display_name="Spin-off share price",
        spreadsheet_column_name="Spin-off share price",
        category="transaction_mechanics",
        description="Spinco share price on first regular-way trading day close.",
        data_type="currency",
        unit="USD",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.MARKET_DATA],
        required_inputs=["regular_way_trading_date"],
        requires_human_review=False,
    ),
    FieldDefinition(
        field_key="spinoff_market_cap",
        display_name="Spin-off market capitalization",
        spreadsheet_column_name="Spin-off market capitalization",
        category="transaction_mechanics",
        description="Spinco market cap = spinoff_share_price x spinoff_shares_outstanding.",
        data_type="currency",
        unit="USD",
        extraction_category=ExtractionCategory.CALCULATED,
        preferred_source_types=[],
        calculation_formula="spinoff_share_price * spinoff_shares_outstanding",
        required_inputs=["spinoff_share_price", "spinoff_shares_outstanding"],
        requires_human_review=False,
    ),
    FieldDefinition(
        field_key="is_100_percent_spinoff",
        display_name="Whether it was a 100% spin-off",
        spreadsheet_column_name="Whether it was a 100% spin-off",
        category="transaction_mechanics",
        description="Whether the parent distributed 100% of spinco shares to shareholders (vs. a partial spin / carve-out IPO retaining a stake).",
        data_type="bool",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10, SourceType.INFORMATION_STATEMENT],
        alternate_source_types=[SourceType.FORM_8K],
        search_terms=["distribute all", "100%", "approximately 100%", "retain no shares", "retain an ownership interest"],
        ai_extraction_instructions="Look for the specific percentage of spinco shares being distributed. Flag as uncertain if the filing describes a phased or partial distribution.",
    ),
    FieldDefinition(
        field_key="parent_retained_ownership",
        display_name="Whether the parent retained ownership",
        spreadsheet_column_name="Whether the parent retained ownership",
        category="transaction_mechanics",
        description="Whether the parent retained any equity stake in spinco post-distribution.",
        data_type="bool",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10, SourceType.INFORMATION_STATEMENT],
        alternate_source_types=[SourceType.FORM_8K],
        search_terms=["retain", "retained interest", "will continue to own"],
        ai_extraction_instructions="This is closely related to is_100_percent_spinoff but not identical — a company can distribute 100% of one class while retaining another. Extract independently and flag if inconsistent with is_100_percent_spinoff.",
    ),
    FieldDefinition(
        field_key="spin_preceded_or_with_parent_acquisition",
        display_name="Whether the spin occurred before or in conjunction with the parent being acquired",
        spreadsheet_column_name="Whether the spin occurred before or in conjunction with the parent being acquired",
        category="transaction_mechanics",
        description="Whether the spin-off happened as a precursor to, or simultaneously with, an acquisition of the parent (a common 'RMT'-style or defensive structure).",
        data_type="bool",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_8K],
        alternate_source_types=[SourceType.PRESS_RELEASE],
        search_terms=["merger agreement", "in connection with the merger", "concurrently with"],
        ai_extraction_instructions="Search 8-Ks filed within +/- 90 days of the spin-off announcement for merger/acquisition language referencing the parent as the target.",
    ),
    FieldDefinition(
        field_key="is_regulatory_driven",
        display_name="Whether the spin-off was regulatory driven",
        spreadsheet_column_name="Whether the spin-off was regulatory driven",
        category="transaction_mechanics",
        description="Whether the spin-off was required or strongly motivated by antitrust/regulatory conditions (e.g. a merger-clearance divestiture).",
        data_type="bool",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10, SourceType.INFORMATION_STATEMENT],
        alternate_source_types=[SourceType.PRESS_RELEASE],
        search_terms=["regulatory approval", "antitrust", "consent decree", "required by", "condition of"],
        ai_extraction_instructions="Distinguish 'regulatory driven' (required as a condition of something) from routine regulatory approvals that any spin-off needs. Only mark true if the filing frames the spin as a regulatory requirement, not a strategic choice.",
    ),
    FieldDefinition(
        field_key="reversed_failed_acquisition",
        display_name="Whether the spin reversed a failed acquisition",
        spreadsheet_column_name="Whether the spin reversed a failed acquisition",
        category="transaction_mechanics",
        description="Whether the spin-off effectively unwinds a previously completed acquisition that did not work out.",
        data_type="bool",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10, SourceType.INFORMATION_STATEMENT],
        alternate_source_types=[SourceType.EARNINGS_CALL_TRANSCRIPT],
        search_terms=["previously acquired", "acquired in", "originally acquired", "combination"],
        ai_extraction_instructions="This usually requires background/history context not stated explicitly as 'this reverses a failed deal' — treat as a judgment field with lower default confidence; require corroboration from the 'background of the spin-off' narrative section.",
        requires_human_review=True,
    ),

    # ─────────────────────────────────────────────────────────────────
    # Capital structure
    # ─────────────────────────────────────────────────────────────────
    FieldDefinition(
        field_key="spinoff_debt",
        display_name="Spin-off debt",
        spreadsheet_column_name="Spin-off debt",
        category="capital_structure",
        description="Total debt spinco is capitalized with at separation (as disclosed in the capitalization table).",
        data_type="currency",
        unit="USD",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.FORM_10],
        alternate_source_types=[SourceType.XBRL],
        structured_extraction_strategy="Form 10 pro forma capitalization table; XBRL 'Debt'/'LongTermDebt' once spinco's first 10-K/10-Q is filed.",
        requires_human_review=True,
        validation_rules=["state whether figure includes or excludes lease liabilities"],
    ),
    FieldDefinition(
        field_key="spinoff_debt_to_ebitda",
        display_name="Spin-off debt-to-EBITDA",
        spreadsheet_column_name="Spin-off debt-to-EBITDA",
        category="capital_structure",
        description="Spinco leverage ratio at separation.",
        data_type="ratio",
        extraction_category=ExtractionCategory.CALCULATED,
        preferred_source_types=[],
        calculation_formula="spinoff_debt / spinoff_ebitda",
        required_inputs=["spinoff_debt", "spinoff_ebitda_margin"],
        requires_human_review=True,
        notes="NOT structured/deterministic: EBITDA is not a standardized XBRL concept, so this ratio inherits ai_assisted uncertainty from its EBITDA input. Corrected from the initial worksheet hypothesis, which mislabeled this 'Structured/Deterministic'.",
    ),
    FieldDefinition(
        field_key="one_time_separation_expenses",
        display_name="One-time separation expenses",
        spreadsheet_column_name="One-time separation expenses",
        category="capital_structure",
        description="Non-recurring costs disclosed as attributable to executing the separation (legal, advisory, systems, rebranding, etc.).",
        data_type="currency",
        unit="USD",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10],
        alternate_source_types=[SourceType.FORM_8K, SourceType.EARNINGS_CALL_TRANSCRIPT],
        search_terms=["separation costs", "one-time costs", "non-recurring", "transaction costs"],
        ai_extraction_instructions="These figures are frequently restated/updated across multiple filings — capture the filing date alongside the value and prefer the most recent pre-distribution estimate.",
    ),
    FieldDefinition(
        field_key="dis_synergy_guidance",
        display_name="Dis-synergy guidance",
        spreadsheet_column_name="Dis-synergy guidance",
        category="capital_structure",
        description="Management's guided estimate of stranded costs / lost scale economies from separating (dis-synergies), typically an annual run-rate figure.",
        data_type="currency",
        unit="USD",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_8K, SourceType.INVESTOR_PRESENTATION],
        alternate_source_types=[SourceType.EARNINGS_CALL_TRANSCRIPT, SourceType.FORM_10],
        search_terms=["dis-synergies", "stranded costs", "dissynergy", "incremental costs"],
        ai_extraction_instructions="Distinguish guidance (forward estimate) from realized/actual figures disclosed after the fact. Store which this is via the 'basis' field on the extracted value.",
        requires_human_review=True,
    ),
    FieldDefinition(
        field_key="dis_synergy_pct_of_prior_year_sales",
        display_name="Dis-synergy guidance as a percentage of prior-year sales",
        spreadsheet_column_name="Dis-synergy guidance as a percentage of prior-year sales",
        category="capital_structure",
        description="dis_synergy_guidance normalized by the parent's prior-year sales.",
        data_type="percent",
        extraction_category=ExtractionCategory.CALCULATED,
        preferred_source_types=[],
        calculation_formula="dis_synergy_guidance / prior_year_sales",
        required_inputs=["dis_synergy_guidance", "last_year_sales"],
        requires_human_review=True,
        notes="Corrected from the initial worksheet hypothesis ('Structured/Deterministic'): the numerator is an AI-assisted value, so the composite cannot be purely deterministic.",
    ),

    # ─────────────────────────────────────────────────────────────────
    # Management and incentives
    # ─────────────────────────────────────────────────────────────────
    FieldDefinition(
        field_key="ceo_came_from_parent",
        display_name="Did the CEO come from the parent?",
        spreadsheet_column_name="Did the CEO come from the parent?",
        category="management_and_incentives",
        description="Whether the spinco CEO at separation was previously an executive at the parent company.",
        data_type="bool",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10],
        alternate_source_types=[SourceType.DEF_14A],
        search_terms=["served as", "prior to the separation", "chief executive officer", "biography"],
        ai_extraction_instructions="Read the Management section biography for the named CEO. True if their prior role was at the parent (any title); false if hired externally at/near separation.",
    ),
    FieldDefinition(
        field_key="ceo_tenure_at_parent",
        display_name="CEO tenure at the parent",
        spreadsheet_column_name="CEO tenure at the parent",
        category="management_and_incentives",
        description="Number of years the spinco CEO worked at the parent company before separation. NOTE: source spreadsheet contains the misspelling 'CEO tenor' — corrected to 'tenure' internally; original label preserved for import/export.",
        data_type="int",
        unit="years",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10],
        alternate_source_types=[SourceType.DEF_14A],
        search_terms=["since", "joined", "has served", "years of experience"],
        ai_extraction_instructions="Compute from stated start date at parent to separation date if an explicit 'X years' figure isn't given. Store both the raw stated text and the computed/normalized number, since these can disagree.",
        required_inputs=["ceo_came_from_parent"],
    ),
    FieldDefinition(
        field_key="cfo_came_from_parent",
        display_name="Did the CFO come from the parent?",
        spreadsheet_column_name="Did the CFO come from the parent?",
        category="management_and_incentives",
        description="Whether the spinco CFO at separation was previously an executive at the parent company.",
        data_type="bool",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10],
        alternate_source_types=[SourceType.DEF_14A],
        search_terms=["chief financial officer", "prior to the separation", "biography"],
        ai_extraction_instructions="Same approach as ceo_came_from_parent, applied to the named CFO.",
    ),
    FieldDefinition(
        field_key="cfo_tenure_at_parent",
        display_name="CFO tenure at the parent",
        spreadsheet_column_name="CFO tenure at the parent",
        category="management_and_incentives",
        description="Number of years the spinco CFO worked at the parent company before separation.",
        data_type="int",
        unit="years",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10],
        alternate_source_types=[SourceType.DEF_14A],
        search_terms=["since", "joined", "has served"],
        ai_extraction_instructions="Same approach as ceo_tenure_at_parent.",
        required_inputs=["cfo_came_from_parent"],
    ),
    FieldDefinition(
        field_key="tsr_based_incentive_plan",
        display_name="TSR-based incentive plan",
        spreadsheet_column_name="TSR-based incentive plan",
        category="management_and_incentives",
        description="Whether spinco management's equity/incentive compensation plan is explicitly tied to total shareholder return (relative or absolute TSR) rather than only time-vesting or non-TSR performance metrics.",
        data_type="bool",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.DEF_14A],
        alternate_source_types=[SourceType.FORM_10],
        search_terms=["total shareholder return", "TSR", "relative TSR", "performance share units", "PSU"],
        ai_extraction_instructions="The first post-spin proxy's Compensation Discussion & Analysis is usually the authoritative source, since Form 10 comp disclosure is often preliminary/incomplete. If only the Form 10 is available at extraction time, mark status as extracted_uncertain and flag for re-check once the first DEF 14A is filed.",
        requires_human_review=True,
    ),

    # ─────────────────────────────────────────────────────────────────
    # Business and financial metrics
    # ─────────────────────────────────────────────────────────────────
    FieldDefinition(
        field_key="last_year_sales",
        display_name="Last-year sales",
        spreadsheet_column_name="Last-year sales",
        category="business_and_financial_metrics",
        description="Spinco's revenue for the fiscal year preceding the spin-off, on a standalone/carve-out basis.",
        data_type="currency",
        unit="USD",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.XBRL, SourceType.FORM_10],
        structured_extraction_strategy="XBRL Revenues / RevenueFromContractWithCustomerExcludingAssessedTax from spinco's first 10-K if available; otherwise Form 10 carve-out financial statements.",
        requires_human_review=True,
        validation_rules=["must confirm this is standalone/carve-out revenue, not a segment-only or parent-consolidated figure"],
    ),
    FieldDefinition(
        field_key="revenue_growth_guidance_current_year",
        display_name="Revenue-growth guidance for the current year",
        spreadsheet_column_name="Revenue-growth guidance for the current year",
        category="business_and_financial_metrics",
        description="Management's guided revenue growth rate for the first fiscal year as a standalone company.",
        data_type="percent",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_8K, SourceType.INVESTOR_PRESENTATION],
        alternate_source_types=[SourceType.EARNINGS_CALL_TRANSCRIPT],
        search_terms=["guidance", "expects revenue growth", "outlook"],
        ai_extraction_instructions="Guidance is often a range; store low/high/midpoint rather than collapsing to one number.",
    ),
    FieldDefinition(
        field_key="spinoff_ebitda_margin",
        display_name="Spin-off EBITDA margin",
        spreadsheet_column_name="Spin-off EBITDA margin",
        category="business_and_financial_metrics",
        description="Spinco EBITDA as a percentage of revenue.",
        data_type="percent",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10],
        alternate_source_types=[SourceType.EARNINGS_CALL_TRANSCRIPT, SourceType.INVESTOR_PRESENTATION],
        search_terms=["EBITDA", "adjusted EBITDA", "EBITDA margin"],
        ai_extraction_instructions="EBITDA is a non-GAAP measure with no standardized XBRL tag — each issuer defines its own adjustments. Extract the company's own stated EBITDA figure AND its reconciliation basis (which add-backs), do not recompute independently unless inputs are fully sourced.",
        requires_human_review=True,
        notes="Corrected from the initial worksheet hypothesis ('Structured/Deterministic') per the brief's explicit instruction not to label EBITDA margin as structured.",
    ),
    FieldDefinition(
        field_key="roic",
        display_name="ROIC",
        spreadsheet_column_name="ROIC",
        category="business_and_financial_metrics",
        description="Spinco return on invested capital, standalone basis.",
        data_type="percent",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10],
        alternate_source_types=[SourceType.EARNINGS_CALL_TRANSCRIPT],
        search_terms=["return on invested capital", "ROIC"],
        ai_extraction_instructions="ROIC has no standardized definition (numerator/denominator choices vary widely). Only extract if the company states its own ROIC figure with a definition; do not calculate from raw XBRL inputs in v1 — flag as a candidate for a defined calculation methodology later, analogous to the TSR methodology model.",
        requires_human_review=True,
        notes="Corrected from the initial worksheet hypothesis ('Structured/Deterministic') per the brief's explicit instruction.",
    ),
    FieldDefinition(
        field_key="parent_division_count",
        display_name="Number of divisions in the parent company",
        spreadsheet_column_name="Number of divisions in the parent company",
        category="business_and_financial_metrics",
        description="Number of reportable segments/divisions the parent operated at the time of the spin-off.",
        data_type="int",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10K],
        alternate_source_types=[SourceType.FORM_10],
        search_terms=["reportable segments", "operating segments", "business segments"],
        ai_extraction_instructions="Use the parent's segment footnote in its 10-K nearest the spin-off date. Count reportable segments as disclosed, not informal business-unit descriptions in prose.",
    ),
    FieldDefinition(
        field_key="parent_sector",
        display_name="Parent sector",
        spreadsheet_column_name="Parent sector",
        category="business_and_financial_metrics",
        description="Parent company's market sector classification.",
        data_type="string",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.MARKET_DATA],
        structured_extraction_strategy="SIC code via EDGAR company facts, or sector field from market data provider (e.g. yfinance).",
        requires_human_review=False,
    ),
    FieldDefinition(
        field_key="parent_industry",
        display_name="Parent industry",
        spreadsheet_column_name="Parent industry",
        category="business_and_financial_metrics",
        description="Parent company's industry classification (finer-grained than sector).",
        data_type="string",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.MARKET_DATA],
        requires_human_review=False,
    ),
    FieldDefinition(
        field_key="spinoff_sector",
        display_name="Spin-off sector",
        spreadsheet_column_name="Spin-off sector",
        category="business_and_financial_metrics",
        description="Spun-off company's market sector classification.",
        data_type="string",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.MARKET_DATA],
        requires_human_review=False,
        notes="Not available until spinco has a live ticker and market-data-provider coverage post-distribution.",
    ),
    FieldDefinition(
        field_key="spinoff_industry",
        display_name="Spin-off industry",
        spreadsheet_column_name="Spin-off industry",
        category="business_and_financial_metrics",
        description="Spun-off company's industry classification.",
        data_type="string",
        extraction_category=ExtractionCategory.STRUCTURED,
        preferred_source_types=[SourceType.MARKET_DATA],
        requires_human_review=False,
    ),

    # ─────────────────────────────────────────────────────────────────
    # Capital allocation
    # ─────────────────────────────────────────────────────────────────
    FieldDefinition(
        field_key="initially_paying_dividend",
        display_name="Initially paying a dividend",
        spreadsheet_column_name="Initially paying a dividend",
        category="capital_allocation",
        description="Whether spinco announced/adopted a dividend policy at or immediately following separation.",
        data_type="bool",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_10],
        alternate_source_types=[SourceType.FORM_8K, SourceType.PRESS_RELEASE],
        search_terms=["dividend policy", "intends to pay", "initial dividend", "does not currently intend to pay"],
        ai_extraction_instructions="Filings often explicitly state the negative ('does not currently intend to pay a dividend') — treat that as a confident False, not 'not found'.",
    ),
    FieldDefinition(
        field_key="dividend_initiated_within_12mo",
        display_name="Initiated a dividend within 12 months",
        spreadsheet_column_name="Initiated a dividend within 12 months",
        category="capital_allocation",
        description="Whether spinco declared its first actual dividend within 12 months of the distribution date, regardless of what was stated at separation.",
        data_type="bool",
        extraction_category=ExtractionCategory.SCHEDULED_MANUAL,
        preferred_source_types=[SourceType.FORM_8K],
        alternate_source_types=[SourceType.FORM_10Q],
        required_observation_window_days=365,
        search_terms=["declared a dividend", "board declared"],
        notes="Requires aggregating all 8-Ks/10-Qs in the 12 months post-distribution — cannot be answered at extraction time for recent spin-offs.",
    ),
    FieldDefinition(
        field_key="preannounced_buyback",
        display_name="Preannounced share buyback",
        spreadsheet_column_name="Preannounced share buyback",
        category="capital_allocation",
        description="Whether spinco announced a share repurchase authorization at or shortly after separation.",
        data_type="bool",
        extraction_category=ExtractionCategory.AI_ASSISTED,
        preferred_source_types=[SourceType.FORM_8K],
        alternate_source_types=[SourceType.FORM_10],
        search_terms=["share repurchase", "buyback authorization", "board authorized"],
    ),

    # ─────────────────────────────────────────────────────────────────
    # Insider behavior
    # ─────────────────────────────────────────────────────────────────
    FieldDefinition(
        field_key="insider_buying_within_3mo",
        display_name="Insider buying within three months of the spin-off",
        spreadsheet_column_name="Insider buying within three months of the spin-off",
        category="insider_behavior",
        description="Whether any officer/director Form 4 filings show open-market purchases within 3 months of distribution.",
        data_type="bool",
        extraction_category=ExtractionCategory.SCHEDULED_MANUAL,
        preferred_source_types=[SourceType.FORM_4],
        required_observation_window_days=90,
        structured_extraction_strategy="Aggregate Form 4 filings for spinco CIK, transaction code 'P' (open market purchase), filed within 90 days of distribution_date.",
        requires_human_review=False,
        notes="Mechanically aggregatable once Form 4 ingestion exists — not a document-reading task, but must wait for the observation window to close for recent spin-offs.",
    ),
    FieldDefinition(
        field_key="cluster_insider_buying_within_3mo",
        display_name="Cluster insider buying within three months",
        spreadsheet_column_name="Cluster insider buying within three months",
        category="insider_behavior",
        description="Whether multiple insiders (typically 3+) bought within a short window of each other, a stronger signal than isolated single-insider buying.",
        data_type="bool",
        extraction_category=ExtractionCategory.SCHEDULED_MANUAL,
        preferred_source_types=[SourceType.FORM_4],
        required_observation_window_days=90,
        calculation_formula="count(distinct insiders with Form 4 code=P within any 30-day sub-window) >= 3",
        required_inputs=["insider_buying_within_3mo"],
        requires_human_review=True,
        notes="Threshold for 'cluster' (count + window) is a judgment call — surface as a configurable parameter, confirm definition with Rich.",
    ),
    FieldDefinition(
        field_key="insider_buyer_count",
        display_name="Number of insiders who bought",
        spreadsheet_column_name="Number of insiders who bought",
        category="insider_behavior",
        description="Count of distinct insiders with an open-market purchase within the 3-month post-spin window.",
        data_type="int",
        extraction_category=ExtractionCategory.SCHEDULED_MANUAL,
        preferred_source_types=[SourceType.FORM_4],
        required_observation_window_days=90,
        required_inputs=["insider_buying_within_3mo"],
        requires_human_review=False,
    ),

    # ─────────────────────────────────────────────────────────────────
    # Performance
    # ─────────────────────────────────────────────────────────────────
    FieldDefinition(
        field_key="tsr",
        display_name="TSR",
        spreadsheet_column_name="TSR",
        category="performance",
        description=(
            "Total shareholder return. NOT a single field — must be computed per a named "
            "methodology configuration (security, start/end date, price convention, dividend "
            "treatment, benchmark). See spinoff_research.tsr_methodology (future module). "
            "This entry models the field's existence in Rich's sheet; it intentionally does not "
            "commit to one methodology."
        ),
        data_type="percent",
        extraction_category=ExtractionCategory.SCHEDULED_MANUAL,
        preferred_source_types=[SourceType.MARKET_DATA],
        required_observation_window_days=365,
        requires_human_review=True,
        notes="Do not implement as a bare percentage. Needs a TSRMethodology config (see Phase 8+ / out of scope for this prototype) before any value is stored.",
    ),
]


FIELD_BY_KEY = {f.field_key: f for f in FIELD_DEFINITIONS}


def get_field(field_key: str) -> FieldDefinition:
    return FIELD_BY_KEY[field_key]


def fields_by_category(category: str) -> List[FieldDefinition]:
    return [f for f in FIELD_DEFINITIONS if f.category == category]


def fields_by_extraction_category(extraction_category: ExtractionCategory) -> List[FieldDefinition]:
    return [f for f in FIELD_DEFINITIONS if f.extraction_category == extraction_category]

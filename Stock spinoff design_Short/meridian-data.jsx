/* global */
/* Meridian-SS data — 35 Greenblatt criteria + LUMN sample state */

const DIMS = [
  { id: "setup",      letter: "S", name: "Setup",            color: "purple", cls: "dim-setup",      key: "Index mechanics & forced selling" },
  { id: "business",   letter: "B", name: "Business Quality", color: "blue",   cls: "dim-business",   key: "Returns on capital and durability" },
  { id: "capital",    letter: "C", name: "Capital Structure",color: "amber",  cls: "dim-capital",    key: "Leverage, pension, maturities" },
  { id: "valuation",  letter: "V", name: "Valuation",        color: "green",  cls: "dim-valuation",  key: "EV/EBIT, P/FCF, sum-of-parts" },
  { id: "incentives", letter: "I", name: "Incentives",       color: "pink",   cls: "dim-incentives", key: "Skin in the game, comp structure" },
];

/* Sample LUMN-spinoff state. Each criterion has:
   - name, sub: criterion text
   - value: extracted value (or null)
   - status: "done" | "partial" | "open" | "blocked"
   - doc: which doc unlocked it (or which is missing/pending)
   - docState: "indexed" | "pending" | "missing" | "skipped"
*/
const CRITERIA = {
  setup: [
    { name: "Market cap",                 value: "$1.6B",        status: "done",    doc: "Form 10 p.12",   docState: "indexed" },
    { name: "Index exclusion",            value: "3 indices",    status: "done",    doc: "Form 10 p.12",   docState: "indexed" },
    { name: "Free float at spin",         value: "94%",          status: "done",    doc: "Form 10 p.18",   docState: "indexed" },
    { name: "Forced selling estimate",    value: "~$410M",       status: "done",    doc: "Form 10 + peers",docState: "indexed" },
    { name: "Institutional ownership",    value: "~62%",         status: "partial", doc: "13F (latest)",   docState: "indexed", note: "Pre-spin estimate; will reset post-distribution" },
    { name: "Strategic rationale",        value: "Articulated",  status: "done",    doc: "Form 10 §1",     docState: "indexed" },
    { name: "Spin dates & ratio",         value: "Jun 14 · 1:4", status: "done",    doc: "Form 10 p.6",    docState: "indexed" },
  ],
  business: [
    { name: "ROIC",                       value: "18.4%",        status: "done",    doc: "XBRL · 10-K",    docState: "indexed" },
    { name: "FCF conversion",             value: "92%",          status: "done",    doc: "XBRL · 10-K",    docState: "indexed" },
    { name: "Revenue growth (5-yr CAGR)", value: "6.4%",         status: "done",    doc: "XBRL · 10-K",    docState: "indexed" },
    { name: "Gross margin trend",         value: "−40 bps YoY",  status: "done",    doc: "XBRL · 10-K",    docState: "indexed" },
    { name: "Competitive moat",           value: "Switching cost", status: "done",  doc: "Form 10 §3 + transcript", docState: "indexed" },
    { name: "Customer concentration",     value: null,           status: "open",    doc: "Form 10 §3.2",   docState: "indexed", note: "Need to extract — section identified" },
    { name: "Operating leverage",         value: "1.4x",         status: "partial", doc: "Q1 transcript",  docState: "indexed", note: "Mgmt commentary only; pending XBRL Q1" },
  ],
  capital: [
    { name: "Net debt",                   value: "$1.2B",        status: "done",    doc: "Form 10 p.85",   docState: "indexed" },
    { name: "Net debt / EBITDA",          value: "3.1x",         status: "done",    doc: "Form 10 p.85",   docState: "indexed" },
    { name: "Pension / OPEB liability",   value: "$340M disclosed", status: "blocked", doc: "Pension footnote", docState: "pending", note: "CFO declined to quantify funding shortfall — footnote expected today" },
    { name: "Debt maturity wall",         value: "2028 ($600M)", status: "done",    doc: "Form 10 p.86",   docState: "indexed" },
    { name: "Interest coverage",          value: "4.8x",         status: "done",    doc: "XBRL · 10-K",    docState: "indexed" },
    { name: "Off-balance-sheet",          value: null,           status: "open",    doc: "Form 10 §6 · 10-K note 18", docState: "indexed" },
    { name: "Capex intensity",            value: "5.2% of sales",status: "done",    doc: "XBRL · 10-K",    docState: "indexed" },
  ],
  valuation: [
    { name: "EV / EBIT",                  value: "9.4x",         status: "done",    doc: "XBRL + Form 10", docState: "indexed" },
    { name: "EV / EBITDA",                value: "6.8x",         status: "done",    doc: "XBRL + Form 10", docState: "indexed" },
    { name: "P / FCF",                    value: "11.2x",        status: "done",    doc: "XBRL + Form 10", docState: "indexed" },
    { name: "Dividend yield",             value: "0%",           status: "done",    doc: "Form 10 §5",     docState: "indexed" },
    { name: "Sum-of-parts upside",        value: null,           status: "open",    doc: "Investor Day",   docState: "indexed", note: "Peer multiples loaded; awaiting your build" },
    { name: "Private market value",       value: null,           status: "open",    doc: "Comparable transactions", docState: "missing" },
    { name: "Mgmt guide vs. consensus",   value: "in-line",      status: "partial", doc: "Q1 transcript",  docState: "indexed" },
  ],
  incentives: [
    { name: "CEO ownership at spin",      value: "0.8% direct",  status: "done",    doc: "DEF 14A · prior", docState: "indexed" },
    { name: "Comp structure",             value: null,           status: "blocked", doc: "DEF 14A · spinco",docState: "missing", note: "Not yet filed on EDGAR — expected post-spin" },
    { name: "Buyback authorization",      value: "$200M (announced)", status: "done", doc: "8-K · May 2",  docState: "indexed" },
    { name: "Insider buying (last 12mo)", value: "None",         status: "done",    doc: "Form 4 history", docState: "indexed" },
    { name: "Parent CEO involvement",     value: "Non-exec chair",status: "done",   doc: "Form 10 §7",     docState: "indexed" },
    { name: "Option vesting cliffs",      value: null,           status: "blocked", doc: "DEF 14A · spinco",docState: "missing" },
    { name: "Capital allocation policy",  value: "FCF buybacks", status: "partial", doc: "Investor Day",   docState: "indexed", note: "Stated philosophy; no formal policy yet" },
  ],
};

/* Dimension-level scores — represent % of research complete
   ≥80 = mostly done · 60-79 = in progress · <60 = just started */
const DIM_STATE = {
  setup:      { score: 100, conf: "Complete" },
  business:   { score: 86, conf: "Mostly done" },
  capital:    { score: 57, conf: "In progress" },
  valuation:  { score: 71, conf: "In progress" },
  incentives: { score: 43, conf: "Just started" },
};
const COMPOSITE = 71;

/* Per-document state used in Documents pane */
const DOCS = [
  { id: "form10",     name: "Form 10",                 type: "10-12B",   pages: 312, method: "EDGAR",     state: "indexed", unlocks: ["S","B","C","V"] },
  { id: "parent10k",  name: "Parent 10-K (CTL)",       type: "10-K",     pages: 218, method: "EDGAR",     state: "indexed", unlocks: ["S","B"] },
  { id: "deck",       name: "Investor Day deck",       type: "PDF",      pages: 47,  method: "Uploaded",  state: "indexed", unlocks: ["B","V"] },
  { id: "q1",         name: "Q1 earnings transcript",  type: "Text",     pages: 31,  method: "Pasted",    state: "indexed", unlocks: ["B","C"] },
  { id: "8k",         name: "8-K · May 2 — buyback",   type: "8-K",      pages: 4,   method: "EDGAR",     state: "indexed", unlocks: ["I"] },
  { id: "pension",    name: "Pension footnote",        type: "10-Q note",pages: null,method: "—",          state: "pending", unlocks: ["C"], blocking: true, note: "Expected today (May 23)" },
  { id: "def14a",     name: "DEF 14A · spinco",        type: "DEF 14A",  pages: null,method: "—",          state: "missing", unlocks: ["I"], blocking: true, note: "Not yet filed — expected post-distribution" },
];

/* Setup dimension metadata — situation type, spin-off structure, indiscriminate selling tracker */
const SETUP_META = {
  situation_type: "Indiscriminate Selling",   // Rich's 4 setups: Demerger Arb / Indiscriminate Selling / Cheap Absolute+Relative / Dividend Catalyst
  spinoff_type:   "100% Spin-off",            // 100% / Partial / Equity Carve-out / Split-off / Reverse Morris Trust
  selling_tracker: {
    shares_outstanding_m: 412,
    cumulative_volume_m:  178,
    days_since_spin:      11,
    pct_traded:           43,    // 178 / 412
    avg_daily_vol_m:      16.2,
    price_trend:          "stabilizing",   // "falling" | "stabilizing" | "recovering"
    note:                 "~40% of shares traded — forced selling near exhaustion per Rich's Module 4 framework (~9 days / ~40% thresholds)",
  },
};

/* Valuation comps table — subject vs. peer multiples */
const VALUATION_COMPS = {
  subject:       { ticker: "LUMN-Spinco",  ev_ebit:  9.4,  ev_ebitda:  6.8,  p_fcf: 11.2 },
  peers: [
    { ticker: "CNSL",  name: "Consolidated Comms", ev_ebit: 14.2, ev_ebitda:  8.9, p_fcf: 16.8 },
    { ticker: "LUMN",  name: "Lumen RemainCo",     ev_ebit:  7.1, ev_ebitda:  5.2, p_fcf:  9.4 },
    { ticker: "WOW",   name: "WideOpenWest",        ev_ebit: 11.8, ev_ebitda:  7.4, p_fcf: 14.1 },
    { ticker: "ATUS",  name: "Altice USA",          ev_ebit: 13.5, ev_ebitda:  8.1, p_fcf: null  },
  ],
  sector_median: { ev_ebit: 11.8, ev_ebitda: 7.4, p_fcf: 14.1 },
};

/* Decision panel — conviction, sizing, exit plan */
const DECISION = {
  composite:          65,
  conviction:         "Medium",          // Low / Medium / High
  setup_type:         "Indiscriminate Selling",
  suggested_position: "5%",              // Rich's framework: 5% medium, 10% high conviction
  rationale:          "Pension footnote unresolved; two I-dimension criteria blocked. Size up to 10% when C dimension clears (footnote + DEF 14A).",
  exit_catalyst:      "Pension footnote disclosure + DEF 14A on EDGAR",
  hold_period:        "12–18 months",
  downside_protection: "Buyback $200M + FCF yield 8.9%",
  status:             "Watch",           // Pass / Watch / Initiate / Full position
};

window.MeridianData = { DIMS, CRITERIA, DIM_STATE, COMPOSITE, DOCS, SETUP_META, VALUATION_COMPS, DECISION };

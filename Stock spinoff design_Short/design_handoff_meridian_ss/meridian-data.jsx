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

/* Dimension-level scores derived (for sample). PRD: ≥80 green, 60-79 amber, <60 red */
const DIM_STATE = {
  setup:      { score: 88, conf: "High", playedOut: true },
  business:   { score: 82, conf: "High" },
  capital:    { score: 45, conf: "Low",  blocking: true },
  valuation:  { score: 60, conf: "Med" },
  incentives: { score: 48, conf: "Low",  blocking: true },
};
const COMPOSITE = 65;

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

window.MeridianData = { DIMS, CRITERIA, DIM_STATE, COMPOSITE, DOCS };

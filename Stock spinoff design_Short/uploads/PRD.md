# Meridian-SS — Product Requirements Document

**Version:** 0.1 (draft)
**Status:** UI scaffold complete (v1) · Backend not yet wired
**Last updated:** 2026-05-23

---

## 1. Problem Statement

Special-situation and spinoff investing is information-dense and process-intensive. An analyst must simultaneously:

- Source the right filings and third-party documents for an obscure, newly-listed company
- Extract and cross-reference 30–40 financial and qualitative metrics across multiple sources
- Stress-test the thesis against structural flaws (pension liabilities, forced selling, governance risks)
- Produce a structured investment memo and log the decision with explicit conviction and risk acknowledgement
- Monitor the position after entry and track whether management delivered on stated commitments

No existing tool handles this end-to-end workflow for special situations specifically. General research tools (Bloomberg, FactSet) lack the workflow layer; general AI chat tools lack grounding in source documents and have no memory of prior decisions.

Meridian-SS is a personal research workstation that automates the mechanical parts of this process while keeping the analyst in the decision seat.

---

## 2. Target User

Solo investor or small fund analyst covering spinoffs and other corporate special situations (carve-outs, post-bankruptcies, risk arbitrage, rights offerings). Comfortable with SEC filings and financial analysis. Not necessarily technical — no coding required to use the product.

---

## 3. Core Workflow — New Analysis (6 Steps)

The primary user flow is a linear wizard that can be revisited at any step.

### Step 1 · Intake
- Enter the spinoff ticker and situation type (Spinoff, Carve-out, Split-off, etc.)
- Enter the parent ticker (optional; auto-detected when possible)
- Optional seed URL (newsletter article, writeup)
- Validate ticker against SEC EDGAR: confirm CIK, company name, state of incorporation, SIC sector
- **Output:** Validated entity record stored in session

### Step 2 · Documents
- **EDGAR bulk fetch:** Pulls Form 10/10-12B, 10-K, 10-Q, 8-K, DEF 14A for spinco and parent in one click
- **Per-document checklist** grouped by entity (Spinoff / Parent / Third party), each showing which metrics it unlocks:
  - Form 10 / 10-12B/A → HTML link or EDGAR
  - Investor day deck → PDF upload only
  - Parent 10-K → HTML link or EDGAR
  - Earnings transcripts (last + prior) → text paste or URL
  - Newsletter / writeup → text paste, URL, or PDF
- Each document type restricts available ingestion methods to what is appropriate (e.g. transcripts cannot be EDGAR-fetched)
- Skipping a document is allowed; soft warning if fewer than 4 documents are added
- **Output:** Document states persisted to SQLite; notes saved for pasted text

### Step 3 · Ingest
- Per-document ingestion pipeline (see Section 6)
- Shows progress row-by-row; errors (403, parse failures) surfaced inline with retry/upload options
- **Output:** Hybrid vector + BM25 index in Chroma; XBRL sidecar and Haiku sidecar stored in SQLite

### Step 4 · Explore (optional)
- Grounded Q&A against the indexed corpus
- Citations shown at the chunk level (document, page/section)
- Save any answer as a note; tag answers for later retrieval
- **Output:** Q&A history persisted in session; saved answers in SQLite notes

### Step 5 · Committee
- **Pre-run coverage check:** Full 35-metric Greenblatt grid shown before the committee runs, color-coded green (data available) / amber (missing). User can navigate back to Step 2 to add documents.
- **5-agent investment committee** debates in 3 rounds:
  - Setup Specialist — index mechanics, forced selling, float
  - Business Quality Analyst — ROIC, FCF, moat, growth
  - Capital Structure Analyst — leverage, pension, maturities
  - Valuation Analyst — EV/EBIT, P/FCF, sum-of-parts
  - Devil's Advocate — structural flaws, disclosure gaps
- Each agent uses XBRL sidecar for numbers and vector retrieval for qualitative context
- Produces dimension scores (0–100) and a composite
- **Output:** Scorecard stored in session; agent summaries displayed

### Step 6 · Memo and Decision
- Claude Opus 4.7 produces a structured 8-section investment memo
- Composite score displayed with 5-dimension breakdown
- **Decision journal** (required before completion):
  - Primary driver, conviction (1–10), biggest acknowledged risk, core thesis (1–3 sentences)
  - Verdict: Invest / Watch / Reject
- **Tendency Coach** shown on re-runs: flags behavioral patterns (#SetupParalysis, #AnalysisParalysis, premature conviction) based on decision history
- **Output:** Memo and decision log persisted to SQLite; situation marked complete

---

## 4. Navigation and Layout

### Primary navigation
- Left sidebar (persistent across all sections): New analysis, Dashboard, Updates, Company detail, Dev
- No page reloads — all navigation via `st.session_state.active_tab`

### New Analysis wizard
- Horizontal step bar at top showing all 6 steps with status (pending / active / done)
- Free navigation between any already-reached step (no re-validation required when returning)
- Data persists when switching steps; documents can be added after the first pass through Step 2

### Dashboard
- Portfolio summary tiles: Invested, Watchlist, Pending review, Avg score
- Portfolio table with 5-dimension score squares per position
- Pending review card for most recent update

### Updates (inbox)
- Quarterly update notifications (10-Q filed)
- 8-K alerts for material events
- Pending document reminders (user-flagged items from prior runs)
- Each card links to re-run workflow for the relevant company

### Company Detail
- Per-company deep-dive with sub-navigation: Overview, Coverage, Memos, Scorecard, Metrics, Promises, Documents, Notes, Q&A
- Coverage tab: same 35-metric grid as Step 5, green/amber color-coded rows, unlock hints showing which document to add
- Notes tab: all saved notes and pasted text for the company
- Documents tab: persisted document list from SQLite

### Dev (testing utility)
- Database inspector: shows all tickers with document and note counts
- Delete-by-ticker button wipes all documents and notes for a company in one click

---

## 5. Data Coverage — Greenblatt 35-Criterion Scorecard

Five dimensions, 7 criteria each. Each criterion is linked to one or more required documents.

| Dimension | Color | Key metrics |
|-----------|-------|-------------|
| Setup | Purple | Market cap, index exclusion, free float, forced selling estimate, institutional ownership, strategic rationale, spin dates |
| Business Quality | Blue | ROIC, FCF conversion, revenue growth, gross margin trend, competitive moat, customer concentration, operating leverage |
| Capital Structure | Amber | Net debt, net debt/EBITDA, pension/OPEB, debt maturity wall, interest coverage, off-balance-sheet, capex intensity |
| Valuation | Green | EV/EBIT, EV/EBITDA, P/FCF, dividend yield, sum-of-parts upside, private market value, mgmt guide vs. consensus |
| Incentives | Pink | CEO ownership, comp structure, buyback authorization, insider buying, parent CEO involvement, option vesting, capital allocation |

Score thresholds: ≥ 80 → green, 60–79 → amber, < 60 → red.

---

## 6. Ingestion Pipeline

### Per-document strategy

| Document | Source | Parsing | Financial data |
|----------|--------|---------|----------------|
| Form 10 / 10-12B | HTML (EDGAR or URL) | BeautifulSoup → H-tag section split → MarkdownHeaderTextSplitter | Haiku sidecar (XBRL not available on registration statements) |
| Parent 10-K / 10-Q | HTML (EDGAR or URL) | BeautifulSoup → H-tag section split | XBRL company facts API (primary); Haiku for pension footnotes, adjusted EBITDA, debt schedules |
| Investor day deck | PDF upload | LlamaParse → section split | Haiku for projection tables |
| Earnings transcript | Text paste or URL | Paragraph split, Q&A blocks intact | Vector index only |
| Newsletter / writeup | Text paste, URL, or PDF | Paragraph split | Vector index only |

### Chunking
- Primary: `MarkdownHeaderTextSplitter` (H1–H3) after HTML extraction
- Fallback: `RecursiveCharacterTextSplitter` (1,200 token max, 150 token overlap) for oversized sections
- All chunks → OpenAI ada-002 embeddings → Chroma (`meridian_ss_{situation_id}`) + BM25 (`rank-bm25`) hybrid index

### Sidecars
- **XBRL sidecar:** `data.sec.gov/api/xbrl/companyfacts/CIK{n}.json` → structured JSON of all GAAP-tagged metrics, stored in SQLite. Committee agents query this directly for standard financial numbers — no vector search.
- **Haiku sidecar:** One Claude Haiku call per targeted HTML section (Form 10 cap table, pension footnote, debt maturity schedule, adjusted EBITDA definition). Stored alongside XBRL in SQLite.

---

## 7. AI Agents and Models

### InvestmentCommittee

5 agents debate in 3 rounds via a `MessageBus`:

| Agent | Model | Dimension |
|-------|-------|-----------|
| Setup Specialist | Claude Sonnet 4.6 | Setup |
| Business Quality Analyst | Claude Sonnet 4.6 | Business |
| Capital Structure Analyst | Claude Sonnet 4.6 | Capital |
| Valuation Analyst | Claude Sonnet 4.6 | Valuation |
| Devil's Advocate | Claude Opus 4.7 | All (adversarial) |

Each agent receives: XBRL sidecar (structured numbers) + vector retrieval results (qualitative context) + prior round summaries.

### Memo Generation
- Claude Opus 4.7
- 8-section structured memo (situation overview, setup quality, business quality, capital structure, valuation, incentives, risks, recommendation)

### Haiku Sidecar
- Claude Haiku 4.5
- Narrow extraction tasks only — not for synthesis or Q&A

---

## 8. Persistence

All data stored in `meridian.db` (SQLite, local):

| Table | Contents |
|-------|----------|
| `documents` | ticker, label, method (edgar/url/uploaded/text/skipped), created_at |
| `notes` | ticker, label, content (pasted text), created_at |
| `decision_logs` | ticker, verdict, conviction, driver, risk, thesis, created_at |
| `xbrl_sidecar` | situation_id, cik, facts_json, fetched_at |
| `haiku_sidecar` | situation_id, doc_key, section, extracted_json, created_at |

API keys (OpenAI, Anthropic) are entered at runtime via the UI and written to `os.environ`. They are not persisted.

---

## 9. Build Phases

### Phase 1 — UI scaffold ✅ (current)
- 6-step wizard with full UI
- EDGAR ticker validation and bulk filing fetch
- Document checklist with per-type method restrictions
- 35-metric coverage grid (color-coded)
- SQLite persistence for documents and notes
- Dev DB inspector

### Phase 2 — Ingestion backend
- Wire BeautifulSoup HTML parser for 10-K/10-Q/Form 10
- Wire LlamaParse for PDF investor decks
- Wire XBRL company facts fetch and SQLite storage
- Wire Haiku sidecar for targeted extraction
- Wire Chroma + BM25 index build

### Phase 3 — Q&A and committee
- Wire Sonnet 4.6 hybrid retrieval Q&A (Step 4)
- Wire InvestmentCommittee 5-agent debate (Step 5)
- Score computation from agent outputs

### Phase 4 — Memo and decision loop
- Wire Opus 4.7 memo generation (Step 6)
- Persist decision logs
- Build Tendency Coach behavioral pattern detection
- Build Updates inbox with 10-Q/8-K monitoring

### Phase 5 — Portfolio and monitoring
- Full Dashboard with real data
- Quarterly re-run workflow
- Promise tracking (Promises tab in Company Detail)
- Export (PDF memo, CSV scorecard)

---

## 10. Out of Scope (v1)

- Multi-user / authentication
- Cloud deployment with persistent storage (currently local SQLite only)
- Real-time price data or market feeds
- Broker integration or order management
- Mobile layout

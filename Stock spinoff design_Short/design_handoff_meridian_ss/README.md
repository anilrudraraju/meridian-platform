# Handoff: Meridian-SS — special-situations research workstation

## Overview

**Meridian-SS** is a personal research workstation for spinoff and special-situation investing. It automates the mechanical parts of analyzing a newly-listed spinco (sourcing filings, extracting financial metrics, running a multi-agent committee, producing a structured memo) while keeping the analyst in the decision seat.

The product is built around a single organizing concept: the **Dimension Workspace**. Five Greenblatt dimensions (Setup / Business Quality / Capital Structure / Valuation / Incentives) are the home screen. Documents flow in and "activate" specific criteria. The analyst lives inside the workspace; everything else (intake, committee, memo, decision, quarterly review) is an action on it.

See `PRD.md` for the full product brief (problem statement, target user, data model, AI agents, persistence layer, build phases).

---

## About the Design Files

The files in this bundle are **design references created in HTML/React** — prototypes that show the intended look and behavior of the Meridian-SS UI. They are not production code to copy directly.

The task is to **recreate these designs in the target codebase's environment** using its established patterns and libraries.

The PRD's "Phase 1 — UI scaffold" target was Streamlit, so these mocks were designed against Streamlit's layout constraints (no fixed sidebars / no absolute positioning / no per-button styling / shallow column nesting). If you implement in Streamlit, the layouts translate directly: every sidebar in the mock is a `st.columns` left rail, every "inline expanded panel" is a `st.expander` or conditional `st.container`, button styling stays uniform with one primary variant via `type="primary"`. If you choose a different framework (Next.js, SwiftUI, etc.), feel free to use richer affordances (overlays, true sidebars).

## Fidelity

**High-fidelity.** Pixel-perfect mockups with final colors, typography, spacing, layout grids, hover/focus states, and interaction flows. The developer should recreate the UI faithfully using the codebase's existing libraries and patterns.

The one explicit creative input the developer should not change: the PRD's dimension color assignments (Setup→Purple, Business→Blue, Capital→Amber, Valuation→Green, Incentives→Pink). Everything else can be adapted to the codebase's design system.

---

## Information Architecture

The product has **four** top-level views, accessed via a left sidebar:

1. **Workspace** — the per-company analysis surface. This is where 80% of usage happens.
2. **Portfolio** — overview of all companies the analyst has opened workspaces for.
3. **Updates** — inbox of state-changing events (10-Qs filed, 8-Ks filed, missing docs landing, promises coming due).
4. **Dev** — local SQLite inspector + API key management.

A fifth path — **New analysis** — is a 30-second intake modal (ticker + parent + situation type) that creates a fresh workspace.

There is **no separate wizard, no separate "Coverage" tab, no separate "Memos/Notes/Q&A" tabs**. Those PRD features collapse into actions and panels on the Workspace itself. This is a deliberate departure from the PRD's "9-tab Company Detail" suggestion: the Dimension Workspace IS the company detail.

---

## Screens

### 1. App shell

**Layout:** 2-column grid · sidebar 220px fixed · main content max-width 1240px.

#### Sidebar (left, full height, sticky)
- Brand block at top: 24px gradient glyph (linear-gradient `var(--d-setup)` → `var(--d-incentives)` 135°) + "Meridian-SS" (14.5px / 600) + "v0.1 · special situations" (11.5px / `--text-3`).
- "+ New analysis" primary button (full-width, centered) opens intake modal.
- Nav list (4 items): Workspace, Portfolio, Updates, Dev. Each row 9px/12px padding, 14px text, icon 16px, optional badge (Updates · 4) or count (Workspace · "LUMN", Portfolio · 7).
- Footer pinned to bottom of sidebar: "API keys connected · 2/2" status pill + DB summary ("meridian.db · 4 tickers · 27 docs").

#### Main column
- 26px top padding, 32px horizontal, 80px bottom.
- Each view renders inside this column.

### 2. Workspace (the heart of the product)

The single page for analyzing one company. For a LUMN spinoff being researched, contains in vertical order:

#### A. Hero header
- 3-column grid: composite ring (76px) · meta block · stats block.
- **Composite ring** is a conic-gradient: `conic-gradient(var(--green) 0 var(--p), rgba(255,255,255,0.08) 0)` where `--p` = `(score × 3.6)deg`. An inner mask circle (6px inset, `--bg-elev` fill) creates the ring. Center text is the integer composite score (22px / 500 / tabular-nums).
- Meta: small caps label (12px / `--text-3`) "LUMN · spinoff from CenturyTel · CIK 0000018926" → company name (17px / 500) → sub-line (13px / `--text-3`).
- Stats (right-aligned, 3 columns, 24px gap): Status (state pill), Last committee (date · version), Memo (version + state).

#### B. Quarterly review banner (conditional)
- Visible when a new 10-Q has been auto-ingested but not yet reviewed.
- Border-left 2px `var(--d-business)`. Document icon · text block · actions.
- Text: "Quarterly review available" small-caps label, "LUMN Q1 2026 10-Q filed May 9 · auto-ingested · **6 criteria moved**, 3 promises reconciled.", subtext "Composite drifted 65 → 56 before committee re-run."
- Actions: "Watching EDGAR" pill (green dot + label) + primary "Review changes →" button.
- Clicking the banner OR the button opens the **Quarterly Review panel** inline.

#### C. Tendency Coach banner (conditional, dismissible)
- Border-left 2px `var(--amber)`. Behavioral pattern callout.
- Example: "You've re-run the committee 3× this week without adding a new document. Pattern detected: **#AnalysisParalysis**. Suggested move: chase the pension footnote — it unblocks the only criterion blocking your Capital score."
- Patterns: #AnalysisParalysis, #SetupParalysis, #PrematureConviction, etc. (PRD §3 Step 6).
- Dismissible (ghost button).

#### D. Next-best-action guide
- Border-left 2px `var(--blue)`. Always visible.
- Names the single highest-leverage move and quantifies the expected score impact.
- Example: "Capital is your weakest dimension. The pension footnote (expected today) unblocks 1 criterion and stabilizes the score. Likely composite move: +6 to +12."
- Actions: "Check EDGAR ↗" secondary + "Upload footnote" primary.

#### E. Primary action bar
Right-aligned row of 4 buttons:
- `+ Add document` — opens add-document flow (EDGAR fetch / upload / paste URL).
- `💬 Ask the corpus` — opens grounded Q&A panel.
- `📋 Memo` — opens memo viewer (or generates if none).
- `Run committee →` (primary) — triggers committee panel.

#### F. Dimension grid (the centerpiece)
5-column CSS grid, 12px gap. Each cell is a **DimCard** with:
- **Top row:** colored letter chip (S/B/C/V/I, 22px square, dimension-tinted) + dimension name + score (28px / 500 / tabular-nums, color-thresholded green ≥80 / amber 60-79 / red <60). If the dimension is "Played" (e.g. Setup after the spin has happened), show the word "Played" instead of a number.
- **Coverage bar:** 7-segment bar (one segment per criterion). Each segment is colored full / partial-opacity / blocked-amber / unset based on the criterion's status. Label above reads "Coverage · 5/7 · 1 blocked".
- **Top-4 criteria preview:** small rows with circular pip (color matches state: done green / partial amber-light / open hollow / blocked amber). Criterion name only (no value at this density).
- **"+3 more criteria"** muted label.
- **Next block** (footer): "Next: <criterion name>" + a small doc chip naming the document that unlocks it. If the doc state is "pending" or "missing", chip is amber.

The whole card is clickable. Clicking expands a **Dimension Detail panel** inline below the grid.

#### G. Dimension Detail panel (inline expansion)
Replaces nothing — appears between the dim grid and the rail below. Slides in with a 14ms fade/translate animation.

- Header: dimension letter + name + key question subtitle + score + confidence label + "Collapse ▲" button.
- **Agent commentary block**: the most recent quote from the relevant committee agent for this dimension (Devil's Advocate often steals the show on Capital).
- **Criteria table** — 5-column grid:
  - Status pip (14px, done/partial/open/blocked with state-specific glyph)
  - Criterion name + optional sub-note
  - Extracted value (tabular-nums; muted/italic when not yet extracted)
  - Doc chip (doc name; amber if pending, red if missing)
  - Action button per row:
    - `blocked` → "Chase doc ↗"
    - `open` → "Extract"
    - `done`/`partial` → "View cite" (ghost)

#### H. Committee panel (conditional, slides in below)
Appears when the user clicks "Run committee →".
- Header: pulsing green dot + "Committee · v3 — round 3 of 3" + actions (Pause, Generate memo, Close).
- 6 agent rows (Setup / Business / Capital / Valuation / Incentives / Devil's Advocate). Each has:
  - Letter avatar tinted with the dim color (Devil's = red)
  - Agent name
  - Body paragraph (the agent's contribution this round)
  - Delta (e.g. "→ 88" flat, "60 → 45" colored red for downward)
  - Ghost "Trace" button (opens the agent's reasoning chain)

#### I. Quarterly Review panel (conditional, inline)
Appears when user clicks the Quarterly banner OR opens the workspace from an Updates quarterly card.
- Header: blue dot + "Quarterly review · Q1 2026 10-Q ingested" + actions (Snooze, Accept changes, **Re-run committee →** primary).
- **Pipeline row**: 5 inline checkmarks for the ingest pipeline (Filing fetched · EDGAR · 64pp / Parsed · 18 sections / XBRL refreshed · 31 facts / Haiku sidecar · pension footnote extracted / Embeddings · 142 new chunks).
- **"Criteria that moved · 6"** section: each row is a 5-col grid: dimension-color dot · criterion name + dim tag · old value · "→ new value" (colored green up / red down) · doc citation.
- **"Promises reconciled · 3"** section: each row has a kept/missed pip + promise text + source + outcome.
- **"Inferred composite"** section: "65 → **56**" with explanatory text.

#### J. Below-grid rail (2-column)
Left column wider (1fr), right 380px fixed.

**Left: Activity feed**
- Header with title + filter chip row (All / Memos / Committee / Decisions / Promises / Docs).
- Each row: type icon (color-tinted per type) · title + meta + body · timestamp.
- Types: memo (blue copy icon), committee (purple user icon), decision (green check), promise (amber bookmark), doc (gray doc), update (gray bell).
- Sample rows: "Memo v2 generated by Opus 4.7 · 8 sections · 2,140 tokens · May 14", "Logged: Watch · conviction 6 · Driver: Setup played; Risk: pension shortfall · May 14", "Mgmt promised SoTP at Q1 investor day · Q1 transcript p.14 · May 6 · Coach: log as a tracked promise."

**Right column** stacks:
- **Documents pane**: header ("Documents · 7" + "+ Add"). Each row is a 5-col grid: file icon (amber clock if pending) · doc name + meta · empty spacer · "Activates" label + S/B/C/V/I dim-chip strip (lit chips for dimensions this doc unlocks) · state label.
- **Decision journal** (inline panel): h ("Decision journal") + 4 fields: Primary driver (text), Conviction (1–10 segmented tick row), Biggest risk (text), Core thesis (textarea). Actions row: "Reject" / "Watch" / "Log invest" (primary).

### 3. Portfolio

A grid of company cards (2 columns).

**Page header**: "Portfolio" h1 + "7 invested · 14 on watchlist · 3 pending review" sub. Right side: "Find ticker" secondary + "+ New analysis" primary.

**4 KPI tiles**: Invested / Watchlist / Pending review (amber) / Avg score.

**Portfolio cards** (2-col grid, 14px gap). Each card is a mini-workspace summary:
- Top row: ticker (bold 16px) + name (gray) + composite score (right-aligned, 22px / color-thresholded).
- 5-segment dim strip (one segment per dimension). Each segment shows letter + score, background-tinted with the dimension color.
- Meta row: state pill (Researching / Memo ready / Invested) + "X blocked · Last activity Y".

Clicking a card opens that ticker's workspace.

### 4. Updates

Inbox of workspace-state deltas. Sorted by recency.

**Page header**: "Updates · workspace deltas" + explanatory sub. Right side: Filter + Mark all read.

**Filter tab row**: All · 4 / Quarterly · 1 / 8-K · 1 / Pending docs · 1 / Promises · 1.

**Update cards** (full-width). Each card has:
- Header row: ticker-tinted icon · title · meta · right-side pill (Review / Ingest / Chase / Track).
- For quarterly updates, an expanded panel below the header shows 5 score tiles (one per dim, color-tinted) with deltas (`−8`, `−15`, etc.) and a summary line.
- For other types, an actions row (Dismiss / Add documents and re-run, etc.).

Clicking a card navigates to that ticker's workspace. Quarterly cards additionally auto-open the Quarterly Review panel.

### 5. New Analysis intake (modal-like)

A centered card on a full-screen `--bg` backdrop. Not a true overlay — replaces the app shell while open (`if (intakeOpen) return <Intake />`).

- Width 520px max.
- "New analysis" pill + "Cancel" ghost button.
- h2 "Open a workspace" + sub "A 30-second intake. Once created, the workspace will guide you through documents, evidence, committee, and memo."
- Fields:
  - **Ticker** (text input)
  - 2-col grid: **Situation type** (select: Spinoff / Carve-out IPO / Split-off / Tracking stock / Post-bankruptcy / Rights offering) + **Parent ticker** ("auto-detected" hint)
  - **Seed thesis URL** (optional)
- "Validate on EDGAR →" primary button → on click, shows a green-tinted success panel: "Validated on EDGAR · {TICKER} · parent {PARENT} · CIK detected · 5 prior filings available for bulk ingest". Button changes to "Open workspace →".
- Footer hint: "You can add documents from the workspace."

### 6. Dev

Page header "Dev · local store" + sub "meridian.db (SQLite) — documents, notes, decision logs, XBRL & Haiku sidecars."

**API keys** section: 2-col grid with masked password inputs for OpenAI and Anthropic. Caption: "Keys are kept in `os.environ` for the session only — not persisted."

**Tickers in store** table: ticker (mono) · documents · notes · coverage · last activity · "Delete {TICKER}" red ghost button.

---

## Design Tokens

### Colors

#### Surfaces
| Token | Value | Usage |
|-------|-------|-------|
| `--bg` | `#0a0a0b` | App background |
| `--bg-elev` | `#131316` | Cards, sidebar, panels |
| `--bg-elev-2` | `#1a1a1f` | Hover surface, active nav item, dim-card focus |
| `--bg-input` | `#1d1d22` | Inputs, selects |

#### Text
| Token | Value | Usage |
|-------|-------|-------|
| `--text` | `#ededee` | Primary text |
| `--text-2` | `#a4a4ad` | Body / secondary |
| `--text-3` | `#6e6e78` | Meta / muted |
| `--text-4` | `#4a4a52` | Disabled / placeholder |

#### Borders
| Token | Value |
|-------|-------|
| `--border` | `rgba(255,255,255,0.06)` |
| `--border-strong` | `rgba(255,255,255,0.10)` |
| `--border-focus` | `rgba(255,255,255,0.18)` |

#### Status colors (for scoring thresholds, alerts, deltas)
| Token | Value | Usage |
|-------|-------|-------|
| `--green` | `oklch(0.78 0.16 148)` | Score ≥80, success, kept-promise |
| `--green-soft` | `oklch(0.78 0.16 148 / 0.14)` | Soft green backgrounds |
| `--amber` | `oklch(0.78 0.15 75)` | Score 60-79, pending, partial |
| `--amber-soft` | `oklch(0.78 0.15 75 / 0.14)` | Soft amber backgrounds |
| `--red` | `oklch(0.70 0.17 25)` | Score <60, missed-promise, error |
| `--red-soft` | `oklch(0.70 0.17 25 / 0.14)` | Soft red backgrounds |
| `--blue` | `oklch(0.72 0.13 245)` | Info, links, primary indicators |
| `--blue-soft` | `oklch(0.72 0.13 245 / 0.16)` | Soft blue backgrounds |

#### Dimension colors (per PRD §5 — do not change)
| Dimension | Letter | Token | Value |
|-----------|--------|-------|-------|
| Setup | S | `--d-setup` | `oklch(0.72 0.15 295)` — purple |
| Business Quality | B | `--d-business` | `oklch(0.72 0.13 240)` — blue |
| Capital Structure | C | `--d-capital` | `oklch(0.78 0.15 75)` — amber |
| Valuation | V | `--d-valuation` | `oklch(0.78 0.16 148)` — green |
| Incentives | I | `--d-incentives` | `oklch(0.72 0.16 350)` — pink |

Each has a `*-soft` variant at `0.16` alpha for chip backgrounds, banner-left accents, and section gradient washes.

### Score thresholds
- Score ≥ 80 → green
- Score 60–79 → amber
- Score < 60 → red
- "Played" (categorical, for Setup post-spin) → no color, text only

### Typography
- Primary: **Geist** (Google Fonts) — weights 400 / 500 / 600.
- Monospace: **Geist Mono** (Google Fonts) — weights 400 / 500. Used for tickers and numerical IDs.
- Sizes used (px): 10.5 (badge), 11–12 (meta/uppercase labels), 13–14 (body), 14.5–15 (emphasized body), 16 (section headings), 18 (panel titles), 20–22 (page titles), 26–32 (scores), 36 (composite hero).
- Letter spacing: `-0.005em` body default, `-0.01em` for 16-18px headings, `-0.02em` for 20px+ titles, `0.04em` for uppercase small-caps labels.
- Numeric data uses `font-variant-numeric: tabular-nums`.

### Spacing
- Card padding: 14-22px depending on density.
- Grid gap: 4px (tight chip rows), 8-12px (most grids), 14-18px (major sections), 22-28px (page section breaks).
- Border radius: 6px (small chips, ticks), 8px (buttons), 10-12px (cards, banners), 14px (intake modal), 999px (pills, dots, score ring).

### Shadows
Minimal use. Cards rely on 1px borders. The only shadow is the subtle `0 1px 0 rgba(255,255,255,0.02) inset` on `.card`.

---

## Interactions & Behavior

### Navigation
- Sidebar nav: click changes `view` state. No URL changes (the prototype is SPA-style; in production, use the codebase's router).
- "+ New analysis" button replaces the entire view with the Intake modal (acts like a route).

### Workspace
- Dim card click → toggles inline detail panel for that dimension (clicking again or another card collapses).
- "Run committee →" → opens Committee panel inline below dim grid.
- "Review changes →" on Quarterly banner → opens Quarterly Review panel inline.
- "+ Add document" → opens document add flow (EDGAR fetch / upload PDF / paste URL — three options in the same surface).
- Tendency Coach banner is dismissible via "Dismiss" ghost button.

### Updates
- Clicking a quarterly card navigates to that ticker's workspace AND auto-opens the Quarterly Review panel.
- Clicking other cards navigates to the workspace.
- Action buttons (Add documents and re-run / Check EDGAR / Upload now / Mark fulfilled) trigger their respective flows.

### Animations
- Panel reveal: 140ms ease, fade + 4px translate.
- Committee "running" dot: 1.4s ease-in-out pulse (opacity 1 ↔ 0.4).
- All other transitions: 120ms ease for hover state changes.

### Hover states
- Buttons darken 1 shade and gain `--border-focus` ring.
- Sidebar items / table rows gain a `rgba(255,255,255,0.03)` background.
- Dim cards gain `--bg-elev-2` background + `--border-strong` border.

---

## State Management

Per the PRD, persistence is **local SQLite** (`meridian.db`). The frontend should expose these views (sketched as Streamlit `session_state` keys; adapt to your framework):

| Key | Type | Notes |
|-----|------|-------|
| `active_view` | str | One of `workspace`, `portfolio`, `updates`, `dev`. |
| `active_ticker` | str \| None | Currently-open workspace. |
| `intake_open` | bool | Renders Intake instead of app shell when true. |
| `quarterly_panel_open` | dict[str, bool] | Per-ticker. Auto-set true when navigating from an Updates quarterly card. |
| `committee_running` | dict[str, bool] | Per-ticker. Controls Committee panel visibility. |
| `dim_focus` | dict[str, str \| None] | Per-ticker. Which dim card is expanded. |
| `coach_dismissed` | dict[str, bool] | Per-ticker. |
| `decision_journal_draft` | dict[str, dict] | Per-ticker. Driver / conviction / risk / thesis. |

Backend tables (PRD §8 — already specified): `documents`, `notes`, `decision_logs`, `xbrl_sidecar`, `haiku_sidecar`. Add two more derived/cached tables:

| Table | Contents |
|-------|----------|
| `criteria_state` | (ticker, dimension, criterion, status, value, source_doc, source_page, citation_id, updated_at) — one row per criterion. Recomputed after every ingest. |
| `committee_runs` | (id, ticker, version, started_at, finished_at, composite, dim_scores_json, agent_outputs_json) — append-only history. |

The Quarterly Review panel reads two snapshots of `criteria_state` (before + after the latest ingest) to compute the diff. Promise reconciliation is a separate `promises` table (PRD-implied, not yet listed): `(id, ticker, what, source_doc, source_page, due_quarter, state, resolved_doc)`. The Quarterly Review panel checks each pending promise against the new 10-Q's content (via vector retrieval) and updates `state`.

---

## Components inventory (for the developer)

The HTML reference files break the UI into these reusable pieces. Map them to your framework's component model.

| Component | File | Responsibility |
|-----------|------|----------------|
| `Sidebar` | `meridian-app.jsx` | Brand + new-analysis CTA + nav + footer |
| `WorkspaceHeader` | `meridian-workspace.jsx` | Composite ring + meta + stats |
| `DimCard` | `meridian-workspace.jsx` | One dimension card (collapsed) |
| `DimDetail` | `meridian-workspace.jsx` | Inline-expanded dimension panel with 7-criteria table |
| `CommitteePanel` | `meridian-workspace.jsx` | Live committee output, 6 agents |
| `QuarterlyReviewPanel` | `meridian-workspace.jsx` | Diff view after 10-Q ingest |
| `DocsPane` | `meridian-workspace.jsx` | Documents list w/ activation chips |
| `ActivityFeed` | `meridian-workspace.jsx` | Filterable history |
| `DecisionJournal` | `meridian-views.jsx` | 4-field journal with conviction ticks |
| `Workspace` | `meridian-views.jsx` | Composes everything above |
| `Portfolio` | `meridian-views.jsx` | KPI tiles + portfolio cards grid |
| `Updates` | `meridian-views.jsx` | Filter tabs + update cards |
| `Intake` | `meridian-views.jsx` | New-analysis modal |
| `Dev` | `meridian-views.jsx` | API keys + DB inspector table |

Shared atoms (`.dim-chip`, `.pill`, `.btn`, `.conf-bar`, `.score-tile`, etc.) are defined in `styles.css` + `meridian.css`.

---

## Files in this bundle

| File | Purpose |
|------|---------|
| `PRD.md` | Product requirements (full spec) |
| `Meridian-SS.html` | Entry HTML for the prototype |
| `meridian-app.jsx` | Sidebar + routing |
| `meridian-views.jsx` | Workspace / Portfolio / Updates / Intake / Dev / DecisionJournal |
| `meridian-workspace.jsx` | DimCard / DimDetail / CommitteePanel / QuarterlyReviewPanel / DocsPane / ActivityFeed |
| `meridian-data.jsx` | Sample LUMN data (35 criteria + 7 documents + dim state) |
| `ui.jsx` | Icon set (lucide-style) + reusable atoms |
| `styles.css` | Base tokens + shared atoms |
| `meridian.css` | Meridian-specific tokens + components |

To preview the design, serve the folder and open `Meridian-SS.html`. The HTML uses React via CDN (no build step). Babel transforms the JSX at runtime.

---

## Assets

The prototype uses no external imagery — only Google Fonts (Geist, Geist Mono) and inline SVG icons. The brand glyph is a CSS-only 24px square with a gradient mask. You can replace it with whatever the production brand is.

---

## What to keep, what to adapt

**Keep faithfully:**
- The Dimension Workspace pattern (5 cards · doc→criterion lineage · inline expansion · next-best-action guidance).
- Dimension color assignments (PRD §5).
- Score thresholds and the green/amber/red language throughout.
- The inbox-driven quarterly review loop (the user should never have to remember to refresh data).
- The Tendency Coach as a first-class UI element (PRD §3 Step 6).
- The Decision Journal's 4 required fields (driver, conviction, risk, thesis) before any Invest/Watch/Reject lock-in.

**Adapt to your stack:**
- Icons → your icon library.
- Fonts → your brand fonts (Geist is a clean default; substitute freely).
- Component primitives (buttons, inputs, cards) → your design system.
- Inline-expansion panels → may become overlays or routed sub-pages depending on framework idioms.
- The "sidebar is a column" Streamlit constraint can be relaxed in any other framework.

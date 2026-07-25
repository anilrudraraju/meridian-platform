# Meridian-SS Design Handoff for Claude Code

This document provides a complete UI/UX design specification for integrating with your backend code.

---

## Overview

**Meridian-SS** is a personal research workstation for spinoff and special-situation investing. The UI is organized around the **Dimension Workspace** pattern — a 5-category analysis framework (Setup, Business, Capital, Valuation, Incentives) that guides analysts through stock evaluation.

**Live design:** `Meridian-SS.html` (fully interactive React prototype)

---

## Architecture

### Data Model

```javascript
// Analysis workspace state
{
  ticker: "LUMN",
  parentCompany: "Lumen Tech",
  situationType: "forced-selling-spinoff",
  
  // Dimension scores (0-100)
  dimensions: {
    setup: { score: 88, status: "done" },      // "done", "open", "blocked", "played"
    business: { score: 82, status: "done" },
    capital: { score: 45, status: "blocked" },
    valuation: { score: 60, status: "done" },
    incentives: { score: 48, status: "done" }
  },
  
  // Weighted composite (recalculates when weights change)
  composite: 65,
  weightedComposite: 62,
  
  // Dimension weights (stored as 0-100, displayed as %)
  weights: {
    setup: 20,
    business: 20,
    capital: 20,
    valuation: 20,
    incentives: 20
  },
  
  // Coverage tracking
  coverage: {
    done: 26,
    total: 35,
    blocked: 4
  },
  
  // Criteria (7 per dimension = 35 total)
  criteria: [
    {
      id: "s1",
      dimId: "setup",
      name: "Spin mechanics",
      value: "Tax-free reorg 368(a)(1)(e)",
      status: "done", // "done", "open", "blocked"
      docState: "resolved", // "resolved", "pending", "missing"
      doc: "Form 10 p.85",
      note: null
    },
    // ... 34 more
  ],
  
  // Documents (indexed by source)
  documents: [
    {
      id: "doc1",
      title: "Form 10 §9.2 — Pension liabilities",
      source: "EDGAR", // EDGAR | PDF | Article | Manual
      date: "2026-05-14",
      activates: ["C2", "C5"], // criterion IDs
      state: "ingested"
    },
    {
      id: "doc8",
      title: "Bernstein: LUMN Forced-Selling Spinoff Deep Dive",
      source: "Article",
      date: "2026-05-20",
      activates: ["S1", "B2", "V3"],
      state: "ingested"
    },
    // ...
  ],
  
  // Decision journal entry (scoped to this stock)
  journal: {
    conviction: 7,
    composite: 65,
    status: "researching", // "researching", "ready", "invested", "closed", "watching"
    drivers: "Forced-selling spinoff...",
    risks: "Index-fund forced selling...",
    checkpoints: [
      { type: "doc", title: "Q2 10-Q", dueDate: "2026-05-23" },
      { type: "call", title: "Investor Day", dueDate: "Q1 2027" }
    ]
  },
  
  // Activity feed (latest events)
  activity: [
    { type: "doc", title: "8-K filed", meta: "EDGAR", time: "May 2", body: "..." },
    { type: "promise", title: "Mgmt promised SoTP", meta: "Q1 call", time: "May 6", body: "..." }
  ]
}
```

### Navigation Structure

```
App
├── Portfolio Hub (default view)
│   └── Active / Closed / Watching tabs
│       └── Card → opens Workspace
│
├── Decision Journal
│   └── List of all analyses by ticker
│       └── Click → see decision details
│
├── + New Analysis (sidebar button)
│   └── Intake modal (ticker + parent + situation type)
│       └── Opens blank Workspace
│
└── Workspace (4-step analysis process)
    ├── 📊 Dimensions tab
    │   ├── 5 DimCards (grid)
    │   ├── Click → DimDetail (inline expand)
    │   │   └── All 7 criteria + table
    │   └── Adjust weights panel (tweak slider set)
    │
    ├── 💬 Ask corpus tab
    │   └── Q&A interface with document context
    │
    ├── 🤖 Committee tab
    │   └── AI decision + reasoning (run button)
    │
    └── 📖 Journal tab
        └── Decision entry details
```

---

## Design Tokens

### Colors (Dark theme, oklch)

| Token | Value | Usage |
|-------|-------|-------|
| `--bg` | `#0a0a0b` | App background |
| `--bg-elev` | `#131316` | Cards, panels |
| `--bg-elev-2` | `#1a1a1f` | Hover, active |
| `--bg-input` | `#1d1d22` | Input fields |
| `--text` | `#ededee` | Primary text |
| `--text-2` | `#a4a4ad` | Secondary |
| `--text-3` | `#6e6e78` | Tertiary |
| `--text-4` | `#4a4a52` | Disabled |
| `--border` | `rgba(255,255,255,0.06)` | Default |
| `--border-strong` | `rgba(255,255,255,0.10)` | Emphasized |
| `--border-focus` | `rgba(255,255,255,0.18)` | Focused |
| **Dimension colors** | | |
| `--d-setup` | `oklch(0.72 0.15 295)` | Purple (S) |
| `--d-business` | `oklch(0.72 0.13 240)` | Blue (B) |
| `--d-capital` | `oklch(0.78 0.15 75)` | Amber (C) |
| `--d-valuation` | `oklch(0.78 0.16 148)` | Green (V) |
| `--d-incentives` | `oklch(0.72 0.16 350)` | Pink (I) |
| **Status colors** | | |
| `--green` | `oklch(0.64 0.20 142)` | ✓ Complete (score ≥80) |
| `--amber` | `oklch(0.74 0.22 67)` | ⚠ In progress (60-79) |
| `--red` | `oklch(0.62 0.25 29)` | ✗ Blocked (<60) |
| `--blue` | `oklch(0.60 0.21 254)` | ℹ Info |

### Typography

- **Font family:** Geist (sans) + Geist Mono (mono) via Google Fonts
- **Display:** 24px / 600 weight
- **Heading:** 16px / 600 weight
- **Label:** 11px / 600 weight, uppercase, 0.5px letter spacing
- **Body:** 13px / 400 weight
- **Small:** 12px / 400 weight
- **Mono:** Geist Mono, 12px / 400 weight

### Spacing & Radii

- **Gap/padding:** 4, 6, 8, 12, 16, 20, 24, 28 (pixels)
- **Radii:** 6px (chip), 8px (button), 10-12px (card), 14px (modal), 999px (pill)

### Animations

- **Hover/active:** 120ms ease
- **Panel reveal:** 140ms ease (keyframe: `panelIn`)
- **Committee live dot:** 1.4s ease-in-out pulse

---

## Components & Patterns

### DimCard (5-card grid on Dimensions tab)

**Props:**
```javascript
{
  dim: { id, name, color, emoji },
  score: 65,
  status: "researching", // done | open | blocked | played
  coverage: { done: 4, total: 7 },
  onFocus: () => {}
}
```

**Visual:**
- Background: dimension color + 8% opacity
- Border: dimension color + 30% opacity
- Large score display (38px)
- "Confidence" label with emoji indicator
- Collapse indicator when focused

### DimDetail (inline expand below grid)

**Shows:**
- All 7 criteria for the dimension
- Table: Status pip | Criterion name | Value | Source (doc chip) | Actions (✏️ 📎 Chase doc/Extract)
- Memo button (if available)
- Add document button

### Table Layout (CSS Grid)

```
gridTemplateColumns: "20px 1fr 100px 110px 120px"
                      ├─ status pip
                      ├─ criterion name (flexible)
                      ├─ value (80px, right)
                      ├─ source (90px, center)
                      └─ actions (120px, right)
```

### Adjust Weights Panel

**Trigger:** "⚙ Adjust weights" button in header

**Content:**
- Title: "Adjust weights"
- 5 sliders (one per dimension)
- Range: 0-100% 
- Colors: dimension colors
- Display: "Setup [████████░░] 20%"
- Updates weighted composite in real-time

### Portfolio Hub

**Tabs:** Active | Closed | Watching

**Card layout per position:**
```
┌─ Ticker (large, bold)
├─ Company name · Composite score (color-coded)
├─ Dim score badges (S/B/C/V/I as pills, color-coded)
├─ Coverage: "26 / 35 · 4 blocked"
├─ Status badge (Researching | Invested | Ready | Watching)
└─ "Open Analysis" button
```

### Decision Journal

**View structure:**
- Left rail: Entry list (ticker, date, company, conviction, composite)
- Right panel: Full entry detail
  - Conviction (scale 1-10)
  - Composite score
  - Status badge
  - Primary driver (prose)
  - Key risks (highlighted box)
  - Monitoring checkpoints (3 items)

### Intake Modal

**Flow:**
1. Ticker input (required)
2. Parent company (optional)
3. Situation type selector (dropdown or radio)
4. "Start analysis" button

---

## State Management

### useTweaks Hook

```javascript
const [t, setTweak] = useTweaks({
  weight_setup: 20,
  weight_business: 20,
  weight_capital: 20,
  weight_valuation: 20,
  weight_incentives: 20,
  // Add other tweaks here
});

// Update single value
setTweak("weight_setup", 25);

// Update multiple values
setTweak({ weight_setup: 25, weight_business: 30 });
```

### Component State Patterns

```javascript
// Workspace-level
const [focused, setFocused] = useState(null); // DimCard focus ID
const [activeTab, setActiveTab] = useState("dims"); // dims | corpus | committee | journal
const [tweaksOpen, setTweaksOpen] = useState(false); // Adjust weights panel
const [committee, setCommittee] = useState(false); // Committee results shown?

// Portfolio-level
const [tab, setTab] = useState("active"); // active | closed | watching
const [ticker, setTicker] = useState(""); // Workspace ticker

// Decision Journal
const [selectedEntry, setSelectedEntry] = useState("LUMN-v3"); // Journal entry ID
```

---

## Integration Checklist

### Backend → Frontend Data Flow

- [ ] Load analysis state from SQLite → populate workspace
- [ ] Save criteria values when user edits → persist to DB
- [ ] Calculate composite score whenever any criterion changes
- [ ] Track document ingestion state (pending/missing/resolved)
- [ ] Manage dimension lock state (blocked criteria = blocked dimension)
- [ ] Store decision journal entries by ticker
- [ ] Persist weight tweaks in localStorage or DB

### Frontend → Backend Triggers

- [ ] "Add document" button → file upload or EDGAR search
- [ ] "Run committee" button → call AI/committee API
- [ ] "Save decision" button → persist journal entry
- [ ] "Extract" button → parse document, populate criterion
- [ ] "Chase doc" button → send reminder (email/notification)

### Pages/Views to Implement

1. **Portfolio Hub** (`/portfolio`)
   - List all analyses with tabs
   - Card → `/workspace?ticker=LUMN` or `/workspace/LUMN`

2. **Workspace** (`/workspace/[ticker]`)
   - Dimensions tab (default)
   - Corpus, Committee, Journal tabs
   - Adjust weights modal (overlay)

3. **Decision Journal** (`/journal`)
   - Entry list (left) + detail (right)
   - Click entry → populate detail

4. **Intake Modal** (modal on Portfolio Hub or standalone route)
   - `/new-analysis` or modal on `/portfolio`

---

## CSS Classes Reference

### Available in meridian.css

```css
.btn              /* Base button style */
.btn.pri          /* Primary (filled) button */
.btn.ghost        /* Tertiary (text-only) button */

.pill             /* Rounded pill badge */
.pill.sm          /* Small pill */

.card             /* Elevated card container */

.score-tile       /* Score display (dimension cards) */
.score-green      /* Score ≥80 */
.score-amber      /* Score 60-79 */
.score-red        /* Score <60 */

.status-pip       /* Status indicator dot */
.status-pip.done
.status-pip.open
.status-pip.blocked
.status-pip.played

.m-dim-grid       /* 5-card dimension grid */
.m-dim-card       /* Individual dimension card */
.m-dim-detail     /* Expanded criteria table */
.m-crit-row       /* Criteria table row */

.m-actions        /* Toolbar (flex row) */
.m-main           /* Main content area */
```

---

## Key Files (Source Code Reference)

| File | Component(s) | Purpose |
|------|-------------|---------|
| `meridian-app.jsx` | `App` | Root, routing, sidebar |
| `meridian-views.jsx` | `Workspace`, `Portfolio`, `DecisionJournal`, `Intake` | View logic |
| `meridian-workspace.jsx` | `DimCard`, `DimDetail`, `CommitteePanel`, `WorkspaceHeader` | Workspace components |
| `meridian-data.jsx` | `DIMS2`, `DIM_STATE`, `SCENARIOS`, `COMPOSITE` | Sample data + state |
| `meridian.css` | (all styles) | Design tokens + component styles |
| `styles.css` | (base tokens) | Global colors, type, shadows |
| `ui.jsx` | Icon set, `TabBar` | Shared UI atoms |

---

## Testing Checklist

- [ ] Portfolio Hub displays 3 tabs (Active/Closed/Watching)
- [ ] Click card → opens Workspace with correct ticker
- [ ] Workspace loads with correct dimension scores
- [ ] Clicking dimension card → expands to show 7 criteria
- [ ] Criteria table displays correctly (columns aligned)
- [ ] "Adjust weights" button → opens modal with colored sliders
- [ ] Adjusting weight → composite score updates
- [ ] Click "Run committee" → shows committee results
- [ ] Decision Journal filters by ticker when opened from Workspace
- [ ] Portfolio Hub buttons navigate correctly
- [ ] All colors match design tokens (no hardcoded hex)

---

## Deployment Notes

- The design is a **React prototype** using inline JSX (Babel transpiled)
- For production: extract components into a React app (Next.js, Vite, etc.)
- CSS tokens are abstracted; light theme can be added via `:root[data-theme="light"]`
- Streamlit port: use `st.columns`, `st.expander` for layouts (respects its constraints)
- Icons are inline SVGs (lucide-style); can be swapped with icon library

---

## Questions for Claude Code?

When integrating this design with your backend:

1. **Data persistence:** Will you use Streamlit session state, SQLite, or both?
2. **Document ingestion:** How will "Add document" handle EDGAR vs. file uploads?
3. **AI/Committee:** What API are you calling for the committee decision?
4. **Real-time updates:** Do criteria updates need to broadcast to other users?
5. **Authentication:** Is this single-user or multi-user workspace?

Include these answers in your prompt to Claude Code for faster integration!

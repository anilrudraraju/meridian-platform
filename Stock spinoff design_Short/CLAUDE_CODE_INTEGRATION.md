# Integration Guide: Meridian-SS Design → Claude Code Backend

## Quick Start for Claude Code

When you prompt Claude Code to build the backend, provide this context:

```
I have a complete UI/UX design for Meridian-SS (a stock analysis workstation).
The design is a fully functional React prototype at Meridian-SS.html.

Key integration points:
1. Data model: Workspace state (ticker, dimensions, criteria, weights, composite score)
2. Navigation: Portfolio Hub → Workspace (4-step analysis) → Decision Journal
3. State management: React hooks (useState, useTweaks for weight persistence)
4. Styling: CSS tokens (colors, spacing, typography)
5. Framework: Built for Streamlit (respects constraint: no fixed sidebars, no absolute positioning)

The design files are:
- Meridian-SS.html — Interactive prototype (see this first!)
- DESIGN_HANDOFF.md — Complete specification
- Design Guide.html — Token reference + component showcase
- meridian-*.jsx — Source components (reference for logic)
- meridian.css, styles.css — Design tokens

I need you to:
1. Extract the data model from the prototype
2. Build API endpoints that match the component props
3. Connect form inputs → state updates → API calls
4. Persist state to SQLite
5. Wire up the document ingestion flow
6. Implement the committee decision endpoint
```

---

## File Reference for Claude Code

### To Understand the Data Model
**Read:** `meridian-data.jsx` (sample LUMN state with all 35 criteria)
**Then:** `DESIGN_HANDOFF.md` → "Data Model" section

### To Understand Navigation
**Read:** `meridian-app.jsx` (view routing)
**Then:** `meridian-views.jsx` (Workspace, Portfolio, Journal components)

### To Understand Component Props
**Read:** `meridian-workspace.jsx` (DimCard, DimDetail, CommitteePanel)
**Structure:** Each component has explicit props that map to backend responses

### To Understand Styling
**Read:** `meridian.css` (component classes)
**Then:** `styles.css` (design tokens)
**Reference:** Colors are `oklch()` — use these exact values, don't invent hex

---

## API Endpoints You'll Need to Build

### 1. Load Analysis Workspace
```
GET /api/workspace/:ticker
Response: { ticker, dimensions, criteria, documents, journal, coverage, composite, weights }
```

### 2. Update Criterion Value
```
PATCH /api/workspace/:ticker/criteria/:criterionId
Body: { value: "18.4%", docState: "resolved" }
Response: { criterion, newDimensionScore, newComposite }
```

### 3. Save Weight Adjustment
```
PATCH /api/workspace/:ticker/weights
Body: { weight_setup: 25, weight_business: 20, ... }
Response: { weights, weightedComposite }
```

### 4. Add Document
```
POST /api/workspace/:ticker/documents
Body: FormData { 
  file or url or text,
  source: "EDGAR" | "PDF" | "Article" | "Manual",
  activates: ["S1", "B2", ...] // which criteria does it address?
}
Response: { document, activatedCriteria }
```

**Source types:**
- **EDGAR** — SEC filing (auto-fetch from EDGAR API)
- **PDF** — File upload (parse with OCR/LLM)
- **Article** — URL (fetch + summarize, auto-suggest criteria via Claude)
- **Manual** — Free-text notes (user types or pastes)

### 5. Run Committee Decision
```
POST /api/workspace/:ticker/committee
Response: { composite: 65, reasoning: "...", recommendation: "..." }
```

### 6. Save Decision Journal Entry
```
POST /api/workspace/:ticker/journal
Body: { conviction: 7, status: "researching", risks: "...", checkpoints: [...] }
Response: { entry }
```

### 7. List Analyses (Portfolio Hub)
```
GET /api/portfolio
Response: [
  { ticker, name, composite, state, dimensions, blocked, last, status: "active|closed|watching" }
  ...
]
```

### 8. Load Journal Entry
```
GET /api/journal/:entryId
Response: { ticker, conviction, composite, status, drivers, risks, checkpoints }
```

---

## State Persistence Strategy

### Client-side (React)
- `useTweaks()` hook → weight adjustments persist to localStorage
- Component state → re-syncs with backend on load
- Form inputs → update local state immediately, POST to API on blur/save

### Server-side (SQLite)
- `analyses` table: ticker, parent_company, situation_type, created_date
- `dimensions` table: analysis_id, dim_id, score, status
- `criteria` table: analysis_id, dim_id, order, name, value, status, doc_id
- `documents` table: analysis_id, title, source, date, content
- `weights` table: analysis_id, setup, business, capital, valuation, incentives
- `journal_entries` table: analysis_id, conviction, composite, status, drivers, risks, checkpoints

---

## Component Props Reference

### DimCard
```javascript
{
  dim: { id: "setup", name: "Setup", color: "oklch(...)", emoji: "📋" },
  score: 88,
  status: "done",  // done | open | blocked | played
  coverage: { done: 4, total: 7 },
  focused: false,
  onFocus: (id) => {}
}
```

### DimDetail
```javascript
{
  dimId: "capital",
  scenario: { /* entire workspace state */ },
  onClose: () => {}
}
```

### Portfolio Card
```javascript
{
  ticker: "LUMN",
  name: "Lumen Spinco",
  composite: 65,
  state: "researching",
  states: { S: 88, B: 82, C: 45, V: 60, I: 48 },
  blocked: 4,
  last: "Today",
  status: "active",
  onOpen: (ticker) => {}
}
```

### Decision Journal Entry
```javascript
{
  id: "LUMN-v3",
  ticker: "LUMN",
  company: "Lumen Spinco",
  date: "Today",
  conviction: 7,
  composite: 65,
  status: "researching",
  drivers: "Forced-selling spinoff...",
  risks: "Index-fund forced selling...",
  checkpoints: [
    { type: "doc", title: "Q2 10-Q", dueDate: "2026-05-23" }
  ]
}
```

---

## Key Design Constraints (Streamlit)

When building for Streamlit, remember:

1. **No fixed sidebars** → use `st.columns([0.2, 0.8])` and rebuild on each interaction
2. **No absolute positioning** → stick to flex/grid layouts
3. **Button styling limited** → use one base style + primary variant
4. **Nesting depth** → max 2 levels of `st.columns`
5. **State management** → use `st.session_state` dict for persistence

If moving to Next.js or Flask later, these constraints can be relaxed.

---

## Testing Checklist for Claude Code

- [ ] Portfolio Hub loads with 3 tabs (Active/Closed/Watching)
- [ ] Clicking portfolio card opens Workspace with correct ticker
- [ ] Workspace loads dimensions with correct scores + colors
- [ ] Clicking dimension card expands to show 7 criteria
- [ ] Adjusting criterion value → API POST → dimension score updates
- [ ] "Adjust weights" sliders → composite recalculates in real-time
- [ ] "Run committee" button → calls API, returns reasoning
- [ ] "Add document" button → file/EDGAR flow → criteria auto-populate
- [ ] Decision Journal filters by ticker when opened from Workspace
- [ ] All weights persist on page refresh (localStorage or DB)
- [ ] No console errors; all API responses have correct structure

---

## Example: Connecting "Adjust Weights"

The slider is already wired in the prototype:

```javascript
const [t, setTweak] = useTweaks(WORKSPACE_TWEAK_DEFAULTS);

// When slider moves:
<TweakSlider 
  label="Setup" 
  min={0} max={100} step={1} 
  value={t.weight_setup}
  onChange={(v) => setTweak("weight_setup", v)}
  dimColor="oklch(0.72 0.15 295)"
/>

// Composite recalculates automatically:
const weightedComposite = Math.round((
  DIM_STATE.setup.score * weights.setup +
  DIM_STATE.business.score * weights.business +
  ...
) / totalWeight);
```

**To add backend persistence**, in Claude Code:

```python
# When weights change in UI, POST to:
@app.post("/api/workspace/{ticker}/weights")
def update_weights(ticker: str, weights: dict):
    # Save to DB
    db.update("weights", weights, where={"ticker": ticker})
    
    # Recalculate composite
    new_composite = calculate_composite(ticker)
    
    return {"weights": weights, "weightedComposite": new_composite}
```

Then wire it up on the frontend:

```javascript
onChange={(v) => {
  setTweak("weight_setup", v);  // Local update
  // POST to backend
  fetch(`/api/workspace/${ticker}/weights`, {
    method: "PATCH",
    body: JSON.stringify({ weight_setup: v })
  }).then(r => r.json()).then(data => {
    // Update composite
  });
}}
```

---

## Next Steps

1. **Give Claude Code this entire document** (from "Quick Start" section)
2. **Show Claude Code the live design:** `Meridian-SS.html`
3. **Point to the data model:** `meridian-data.jsx`
4. **Ask for the endpoints above** — one at a time is fine
5. **Test each endpoint** by triggering the UI action
6. **Iterate** — design is flexible; if backend needs changes, let me know

---

## Questions to Answer Before You Start

1. **Streamlit or Flask/Next.js?** (Design assumes Streamlit constraints)
2. **SQLite or cloud DB?** (Affects persistence strategy)
3. **AI/Committee engine?** (API endpoint or local function?)
4. **Document ingestion:** EDGAR feed only, or also file upload + manual?
5. **Single-user or multi-user?** (Affects session management)

---

Good luck! The design is complete and ready to hook up. Let me know if Claude Code needs clarification on any component.

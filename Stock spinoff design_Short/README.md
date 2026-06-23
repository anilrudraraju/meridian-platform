# Meridian-SS — Complete Design & Prototype Package

## What's Included

This package contains the **complete UI/UX design** for Meridian-SS, a research workstation for spinoff and special-situation investing.

### 📊 Core Design Files

- **`Meridian-SS.html`** ← **START HERE** — Fully interactive React prototype (all screens, colors, interactions)
- **`Design Guide.html`** — Design system reference (colors, typography, components)

### 📝 Documentation

- **`DESIGN_HANDOFF.md`** — Complete specification (data model, components, API, checklist)
- **`CLAUDE_CODE_INTEGRATION.md`** — Backend integration guide (endpoints, state management, examples)
- **`UPDATED_NOTES.md`** — Article/Newsletter document source type
- **`CLAUDE.md`** — Project context & conventions

### 💻 Source Code (React Components)

- `meridian-app.jsx` — Root app + sidebar routing
- `meridian-views.jsx` — Workspace, Portfolio, Decision Journal, Intake components
- `meridian-workspace.jsx` — Dimension cards, detail panels, committee, headers
- `meridian-data.jsx` — Sample LUMN spinoff data (35 criteria, 7 docs)
- `meridian.css` — Component styles & design tokens
- `styles.css` — Base colors, typography, spacing
- `ui.jsx` — Icon set & shared components

### 📋 Source Material

- `uploads/PRD.md` — Original product specification
- `uploads/Screenshot*.png` — Reference screenshots from discovery

---

## How to Use

### For Designers / Stakeholders
1. Open **`Meridian-SS.html`** in a browser
2. Click through all screens (Portfolio Hub → Workspace → Decision Journal)
3. Adjust weights, expand criteria cards, see all interactions

### For Developers (Claude Code)

1. **Read the design:** Open `Meridian-SS.html` (5 min)
2. **Understand data:** Read `DESIGN_HANDOFF.md` → "Data Model" section (10 min)
3. **Know what to build:** Read `CLAUDE_CODE_INTEGRATION.md` → "API Endpoints" (15 min)
4. **Reference components:** Explore `meridian-*.jsx` files (see how state flows)
5. **Match colors/tokens:** Use `Design Guide.html` for exact values

### To Give to Claude Code

```
I have a complete UI/UX design for Meridian-SS.

See the interactive prototype: Meridian-SS.html

To integrate with backend:
1. Read DESIGN_HANDOFF.md (data model + components)
2. Build endpoints in CLAUDE_CODE_INTEGRATION.md
3. Connect props in meridian-*.jsx files
4. Use exact color tokens from Design Guide.html

Start with /api/portfolio and /api/workspace/:ticker endpoints.
```

---

## Quick Architecture Overview

### Navigation Flow
```
Portfolio Hub (default)
  ├─ Active / Closed / Watching tabs
  └─ Click card → Workspace

Workspace (4-step process)
  ├─ 📊 Dimensions (5 cards, 35 criteria)
  ├─ 💬 Ask corpus (Q&A with documents)
  ├─ 🤖 Committee (AI decision)
  └─ 📖 Journal (decision details)

Decision Journal
  └─ Entry list → click to see details

+ New Analysis (button)
  └─ Intake modal → blank Workspace
```

### Data Model
```javascript
Workspace {
  ticker: "LUMN",
  dimensions: { setup, business, capital, valuation, incentives },
  criteria: [ 7 per dimension = 35 total ],
  documents: [ EDGAR, PDF, Article, Manual ],
  composite: 65,
  weights: { setup: 20, business: 20, ... },
  journal: { conviction, status, risks, checkpoints }
}
```

### Document Sources
- **EDGAR** — SEC filings (auto-fetch)
- **PDF** — File upload
- **Article** — URL (newsletter, research report)
- **Manual** — Free-text notes

---

## Design System

### Colors (oklch)
- **Setup (S)** — Purple `oklch(0.72 0.15 295)`
- **Business (B)** — Blue `oklch(0.72 0.13 240)`
- **Capital (C)** — Amber `oklch(0.78 0.15 75)`
- **Valuation (V)** — Green `oklch(0.78 0.16 148)`
- **Incentives (I)** — Pink `oklch(0.72 0.16 350)`

### Status Colors
- **Done (≥80)** — Green `#10b981`
- **In Progress (60-79)** — Amber `#d99655`
- **Blocked (<60)** — Red `#ef4444`

### Typography
- Font: Geist (sans) + Geist Mono (mono) from Google Fonts
- Scale: 11px (label) → 24px (display)

### Spacing & Radii
- Gap: 4, 6, 8, 12, 16, 20, 24, 28px
- Radii: 6px (chip), 8px (button), 10-12px (card), 14px (modal)

---

## Key Features

✅ **Dimension Workspace** — 5 Greenblatt dimensions as primary nav (not wizard steps)
✅ **Criteria Management** — 7 criteria per dimension, track status + sources
✅ **Document Ingestion** — EDGAR, PDF, Article, Manual (with auto-suggest for criteria)
✅ **Dynamic Weighting** — Adjust dimension weights, composite recalculates in real-time
✅ **Decision Journal** — Track conviction, risks, monitoring checkpoints per stock
✅ **Corpus Q&A** — Ask questions, search across all documents
✅ **Committee Decision** — AI synthesis of all criteria (run button)
✅ **Portfolio Hub** — Browse all analyses by status (Active/Closed/Watching)

---

## Streamlit Constraints (if porting to Streamlit)

The design respects Streamlit's limitations:
- No fixed sidebars → use `st.columns`
- No absolute positioning → flex/grid only
- Button styling limited → one base style + primary variant
- Column nesting max 2 levels → keep layout simple

If moving to Flask/Next.js later, these can be relaxed.

---

## Next Steps

1. **Review prototype** — Open `Meridian-SS.html`, click through
2. **Ask questions** — Check DESIGN_HANDOFF.md or CLAUDE_CODE_INTEGRATION.md
3. **Build backend** — Use API endpoints in CLAUDE_CODE_INTEGRATION.md
4. **Connect frontend** — Wire component props to your endpoints
5. **Test** — Use checklist in CLAUDE_CODE_INTEGRATION.md

---

## Questions?

**For design clarity:** See DESIGN_HANDOFF.md
**For backend integration:** See CLAUDE_CODE_INTEGRATION.md
**For token values (colors, spacing):** See Design Guide.html
**For component logic:** Check meridian-*.jsx files

---

**Design created:** June 2026
**Framework:** React (prototype) → Streamlit (Phase 1 production)
**Author:** Meridian Design Team

# Meridian-SS — Project Context

Persistent notes so any future chat in this project can resume without losing context. Read this first.

---

## What this project is

**Meridian-SS** — a personal research workstation for **spinoff and special-situation investing**.

The user is building a Streamlit application (Phase 1 per their PRD) backed by SQLite. This project holds the **design references** for that app — high-fidelity HTML/React mockups that show the intended look and behavior.

The PRD lives at `uploads/PRD.md`. Read it before doing meaningful design work.

---

## Where we landed (current state)

**Final design = `Meridian-SS.html`.** Open this first if asked about "the design".

The product is organized around a **Dimension Workspace** pattern:

- 5 Greenblatt dimensions (Setup / Business / Capital / Valuation / Incentives) are the home screen, not steps in a wizard.
- Documents flow in and "activate" specific criteria (each doc has S/B/C/V/I chips showing which dims it unlocks).
- Quarterly 10-Q ingestion is **inbox-driven** — Meridian auto-watches EDGAR; when a filing lands, the user sees a banner on the workspace AND a card in Updates. Clicking either opens the **Quarterly Review panel** (an inline diff: criteria that moved + promises reconciled + inferred composite drift), then a single "Re-run committee" button.
- All PRD features fold into the workspace as **actions and panels**, not separate screens. There is no separate Coverage/Memos/Notes tab.

**Top-level views (sidebar):**
1. **Workspace** — the analysis surface (where 80% of usage happens)
2. **Portfolio** — list of all opened workspaces
3. **Updates** — inbox of state-changing events
4. **Dev** — local SQLite inspector + API keys

Plus **New Analysis** — a 30-second intake modal (ticker + parent + situation type).

---

## Dimension colors (per PRD §5 — do not change)

| Dim | Letter | Color | Token |
|-----|--------|-------|-------|
| Setup | S | Purple | `--d-setup` `oklch(0.72 0.15 295)` |
| Business Quality | B | Blue | `--d-business` `oklch(0.72 0.13 240)` |
| Capital Structure | C | Amber | `--d-capital` `oklch(0.78 0.15 75)` |
| Valuation | V | Green | `--d-valuation` `oklch(0.78 0.16 148)` |
| Incentives | I | Pink | `--d-incentives` `oklch(0.72 0.16 350)` |

Score thresholds: ≥80 green · 60-79 amber · <60 red. "Played" is a categorical value (no color) for Setup post-spin.

---

## Streamlit constraints the user flagged

These shaped every layout decision. Re-check any new design against them:

1. No true fixed sidebars (we fake with `st.columns`)
2. Can't overlap elements or use absolute positioning
3. Buttons can't be styled individually without CSS hacks → keep one base button style + one primary variant
4. Column nesting works but gets narrow quickly → max 2 levels

If user moves to a different framework later, these can be relaxed.

---

## File map

### Final design (Meridian-SS — use this)
| File | Purpose |
|------|---------|
| `Meridian-SS.html` | Entry HTML for the final design |
| `meridian-app.jsx` | Sidebar + view routing |
| `meridian-views.jsx` | Workspace / Portfolio / Updates / Intake / Dev / DecisionJournal |
| `meridian-workspace.jsx` | DimCard / DimDetail / CommitteePanel / QuarterlyReviewPanel / DocsPane / ActivityFeed / WorkspaceHeader |
| `meridian-data.jsx` | Sample LUMN spinoff state — 35 criteria, 7 docs, dim scores |
| `meridian.css` | Meridian tokens + components |

### Shared atoms (used by everything)
| File | Purpose |
|------|---------|
| `styles.css` | Base tokens (colors, type, spacing, shadows) + shared primitives (.btn, .pill, .card, .kpi, .score-tile, etc.) |
| `ui.jsx` | Icon set (`I.doc`, `I.alert`, `I.arrowR`, etc. — lucide-style inline SVGs) + `TabBar`, `Dims`, `Icon` |

### Earlier explorations (kept for history — usually don't touch)
| File | What it was |
|------|-------------|
| `Spinoff Workbench.html` + `app.jsx` + `screens.jsx` + `wizard.jsx` | v1 prototype — 4-tab layout with a 6-step wizard. Replaced. |
| `Spinoff Workbench — alternatives.html` + `alternatives.jsx` + `alternatives.css` | Three direction options (Dimension Workspace · Evidence Board · Guided Co-pilot). User picked Dimension Workspace → became Meridian-SS. |
| `design-canvas.jsx` | Starter component used to present the 3 alternatives side-by-side. |

### Handoff bundle (for Claude Code)
| File | Purpose |
|------|---------|
| `design_handoff_meridian_ss/` | Standalone folder with PRD + all design source + comprehensive README documenting IA, screens, tokens, interactions, state, components. Self-contained — give this to a dev. |

### Source material
| File | Purpose |
|------|---------|
| `uploads/PRD.md` | The product spec. Always reference this. |
| `uploads/Screenshot 2026-05-22 *.png` | User's original screenshots that v1 was built against. |

---

## Design system summary

**Fonts:** Geist (sans) + Geist Mono (mono) via Google Fonts.

**Surfaces:** layered dark
- `--bg` `#0a0a0b` (app)
- `--bg-elev` `#131316` (cards)
- `--bg-elev-2` `#1a1a1f` (hover, focused dim card)
- `--bg-input` `#1d1d22`

**Text:** `--text` `#ededee` → `--text-2` `#a4a4ad` → `--text-3` `#6e6e78` → `--text-4` `#4a4a52`

**Borders:** `rgba(255,255,255,0.06)` → `0.10` strong → `0.18` focus

**Status:** `--green/amber/red/blue` defined in oklch with `*-soft` 14-16% alpha variants for chip backgrounds.

**Radii:** 6 (chip), 8 (button), 10-12 (card), 14 (intake modal), 999 (pill).

**Animation:** 120ms ease for hover, 140ms ease for panel reveals. One ongoing animation: the committee "live" dot pulse (1.4s ease-in-out).

---

## User preferences observed

- Wants UI that **guides the analyst**, not procedural step-by-step UI. UI should act as a checklist AND show what each document activates.
- Likes inline expansion over modals/overlays (also satisfies the Streamlit constraint).
- Wants doc → criterion lineage to be visible (every criterion shows its citing document; every doc shows which dimensions it activates).
- Picked Dimension Workspace direction over Evidence Board and Guided Co-pilot.
- Asked for the design to fit PRD features INTO the workspace pattern, not invent new screens.

---

## Open / possible next moves

If the user picks back up, these are natural next steps (in priority order):

1. **Streamlit port** — translate `meridian-*.jsx` into `app.py` + `pages/` using `st.columns` for layout, `st.expander` or conditional `st.container` for inline panels, `st.session_state` for the state map sketched in `design_handoff_meridian_ss/README.md`.
2. **Add Document add-flow surface** — the "+ Add document" button currently has no destination panel. Design it: EDGAR fetch / Upload PDF / Paste URL — same 3-up layout from the earlier wizard's Step 2.
3. **Memo viewer** — the "📋 Memo" button has no surface yet. Design an inline panel that opens between the dim grid and the rail; should support v1/v2/v3 history toggle and per-section regeneration.
4. **Ask-the-corpus panel** — grounded Q&A surface; was prototyped in earlier wizard Step 4, needs to be folded into the workspace.
5. **Promise tracker detail** — promises currently surface as feed rows. Could grow its own inline panel listing all tracked promises with due quarters.
6. **Portfolio empty state + filtering** — design what an empty portfolio looks like + add filter chips (state, dim score range, situation type).
7. **Light theme** — currently dark only. Tokens are abstracted, so a `:root[data-theme="light"]` block could provide a parallel scheme.

---

## Conventions for future edits

- Always edit `meridian-*.jsx` / `meridian.css` for the final design. Don't touch `app.jsx` / `screens.jsx` / `wizard.jsx` (v1).
- Use the design tokens — never hardcode hex/oklch values inside components.
- Buttons: one base `.btn` + `.btn.pri` for primary + `.btn.ghost` for tertiary. No other variants.
- Inline panels reveal with `panelIn` keyframes — already defined.
- New criteria/data → add to `meridian-data.jsx`.
- Keep the Streamlit constraints in mind on any layout change.
- Verify with `done` + `fork_verifier_agent` after meaningful changes.

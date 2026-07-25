"""
Special Situations Lab — Spinoff Research Lab
Standalone Streamlit module; rendered from app.py via:
    if st.session_state.active_layer == "spinoff_lab":
        spinoff_lab.render()
"""
import os
import json
from datetime import date, datetime
from typing import List

import pandas as pd
import streamlit as st
from core.cost import daily_spend, DAILY_CAP_USD, read_log

from spinoff.models import (
    SpinoffEvent, GreenblattScore, ManagementPromise, ThesisEntry, CostEntry,
)
from spinoff.greenblatt_scorecard import score_spinoff, tier_label, CRITERIA
from spinoff.promise_tracker import load_promises, save_promises, add_promise
from spinoff.thesis_tracker import load_theses, save_theses, add_thesis
from spinoff.cost_tracker import load_costs, log_cost, total_by_category, CATEGORIES

# ── paths ──────────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "spinoffs")
_WATCHLIST_CSV   = os.path.join(_DATA_DIR, "sample_spinoff_watchlist.csv")
_PROMISES_CSV    = os.path.join(_DATA_DIR, "sample_management_promises.csv")
_THESIS_CSV      = os.path.join(_DATA_DIR, "sample_thesis_tracker.csv")
_COSTS_CSV       = os.path.join(_DATA_DIR, "sample_cost_log.csv")

# ── session-state keys ─────────────────────────────────────────────────────────
_SS_SCORES    = "spinoff_scores"       # List[GreenblattScore]
_SS_WATCHLIST = "spinoff_watchlist"    # pd.DataFrame
_SS_PROMISES  = "spinoff_promises"     # List[ManagementPromise]
_SS_THESES    = "spinoff_theses"       # List[ThesisEntry]
_SS_COSTS     = "spinoff_costs"        # List[CostEntry]


def _init_state() -> None:
    if _SS_SCORES not in st.session_state:
        st.session_state[_SS_SCORES] = []
    if _SS_WATCHLIST not in st.session_state:
        try:
            st.session_state[_SS_WATCHLIST] = pd.read_csv(_WATCHLIST_CSV)
        except Exception:
            st.session_state[_SS_WATCHLIST] = pd.DataFrame()
    if _SS_PROMISES not in st.session_state:
        st.session_state[_SS_PROMISES] = load_promises(_PROMISES_CSV)
    if _SS_THESES not in st.session_state:
        st.session_state[_SS_THESES] = load_theses(_THESIS_CSV)
    if _SS_COSTS not in st.session_state:
        st.session_state[_SS_COSTS] = load_costs(_COSTS_CSV)


# ── tier badge helpers ─────────────────────────────────────────────────────────
_TIER_COLOR = {"Deep Dive": "🟢", "Watchlist": "🟡", "Pass": "🔴"}


def _tier_badge(tier: str) -> str:
    return f"{_TIER_COLOR.get(tier, '⚪')} **{tier}**"


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Spinoff Setup
# ══════════════════════════════════════════════════════════════════════════════
def _tab_setup() -> None:
    st.subheader("📋 Spinoff Watchlist")

    df: pd.DataFrame = st.session_state[_SS_WATCHLIST]
    if df.empty:
        st.info("No spinoffs loaded yet. Add one below or upload a CSV.")
    else:
        st.dataframe(df, use_container_width=True)

    st.divider()
    st.markdown("#### Add a Spinoff")
    with st.form("add_spinoff_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        ticker      = c1.text_input("Spinco Ticker *", placeholder="e.g. GEV")
        parent      = c2.text_input("Parent Ticker *", placeholder="e.g. GE")
        company     = c1.text_input("Company Name *",  placeholder="e.g. GE Vernova")
        parent_name = c2.text_input("Parent Name",     placeholder="e.g. General Electric")
        c3, c4, c5  = st.columns(3)
        spin_date   = c3.date_input("Spinoff Date", value=date.today())
        exchange    = c4.selectbox("Exchange", ["NYSE", "NASDAQ", "OTC", "Other"])
        sector      = c5.selectbox("Sector", [
            "Industrials", "Healthcare", "Technology", "Financials",
            "Energy", "Consumer", "Materials", "Media", "Other",
        ])
        cik         = st.text_input("CIK (optional)", placeholder="SEC CIK number")
        notes       = st.text_area("Notes", height=80)
        submitted   = st.form_submit_button("Add to Watchlist", type="primary")

    if submitted:
        if not ticker or not parent or not company:
            st.error("Ticker, Parent Ticker, and Company Name are required.")
        else:
            new_row = {
                "ticker": ticker.upper(), "parent_ticker": parent.upper(),
                "company_name": company, "parent_name": parent_name,
                "spinoff_date": spin_date.isoformat(), "exchange": exchange,
                "sector": sector, "cik": cik, "notes": notes,
            }
            st.session_state[_SS_WATCHLIST] = pd.concat(
                [df, pd.DataFrame([new_row])], ignore_index=True
            )
            st.success(f"✅ Added {ticker.upper()} to watchlist.")
            st.rerun()

    st.divider()
    st.markdown("#### Upload Watchlist CSV")
    uploaded = st.file_uploader("Upload spinoff watchlist CSV", type="csv",
                                key="watchlist_upload")
    if uploaded:
        try:
            st.session_state[_SS_WATCHLIST] = pd.read_csv(uploaded)
            st.success("✅ Watchlist loaded from CSV.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SEC Documents
# ══════════════════════════════════════════════════════════════════════════════
def _tab_sec_documents() -> None:
    st.subheader("📄 SEC Documents")
    st.info(
        "Use **Layer 3 — Document RAG** to fetch and index SEC filings for spinoff companies. "
        "Enter the spinco ticker there (e.g. GEV, SOLV) and choose filing type **Form 10** "
        "for registration statements or **10-K** for annual reports."
    )

    st.markdown("#### Quick Reference — EDGAR Filing Types")
    st.markdown("""
| Filing Type | When Filed | What to Look For |
|-------------|------------|-----------------|
| **Form 10 (10-12B)** | Before trading begins | Standalone financials, spin rationale, carve-out adjustments |
| **10-K** | First annual report | True standalone P&L, mgmt discussion, risk factors |
| **Form S-1** | If IPO structure | Pro-forma financials, use of proceeds |
| **DEF 14A (Proxy)** | Before vote | Executive comp, insider ownership, board composition |
| **SC 13G/D** | After filing | Institutional ownership; track forced sellers exiting |
""")

    st.markdown("#### Checklist — What to Read in a Form 10")
    checklist_items = [
        "Business description and competitive positioning",
        "Carve-out vs. full standalone financials",
        "Related-party transactions with parent (are they arm's-length?)",
        "TSA (Transition Service Agreement) terms and end dates",
        "Stranded costs quantified in MD&A",
        "Management team — did they come from parent or outside hire?",
        "Insider stock grants and vesting schedule",
        "Capital structure post-spin: debt load and covenants",
        "First-time dividend or buyback policy stated",
    ]
    for item in checklist_items:
        st.checkbox(item, key=f"checklist_{item[:20]}")

    st.divider()
    st.markdown("#### Document Notes")
    ticker_note = st.text_input("Ticker", placeholder="e.g. GEV",
                                key="sec_doc_ticker")
    doc_note    = st.text_area("Research Notes (saved to session)",
                               height=150, key="sec_doc_notes")
    if st.button("Save Notes to Session", key="sec_doc_save"):
        key = f"sec_notes_{ticker_note.upper()}"
        st.session_state[key] = doc_note
        st.success(f"Notes saved for {ticker_note.upper()} (session only).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Greenblatt Screen
# ══════════════════════════════════════════════════════════════════════════════
def _tab_greenblatt() -> None:
    st.subheader("🎯 Greenblatt Spinoff Scorecard")
    st.caption(
        "Score each criterion 0–5. "
        "Total ≥ 35 → Deep Dive · 25–34 → Watchlist · < 25 → Pass"
    )

    with st.form("greenblatt_form"):
        c1, c2 = st.columns(2)
        ticker  = c1.text_input("Ticker *", placeholder="e.g. GEV")
        company = c2.text_input("Company Name *", placeholder="e.g. GE Vernova")

        st.markdown("---")
        st.markdown("**Score each criterion 0 (worst) → 5 (best)**")

        scores = {}
        for field_name, label, description in CRITERIA:
            col_label, col_slider = st.columns([2, 3])
            col_label.markdown(f"**{label}**")
            col_label.caption(description)
            scores[field_name] = col_slider.slider(
                label, min_value=0, max_value=5, value=3,
                key=f"gb_{field_name}", label_visibility="collapsed"
            )

        notes     = st.text_area("Analyst Notes", height=80)
        submitted = st.form_submit_button("Score Spinoff", type="primary")

    if submitted:
        if not ticker or not company:
            st.error("Ticker and Company Name are required.")
        else:
            try:
                result = score_spinoff(
                    ticker=ticker.upper(), company_name=company, notes=notes,
                    **scores
                )
                st.session_state[_SS_SCORES].append(result)
                tier = result.tier
                st.markdown(f"### Result: {_tier_badge(tier)}")
                st.metric("Total Score", f"{result.total} / 35")

                col_metrics = st.columns(len(CRITERIA))
                for i, (field_name, label, _) in enumerate(CRITERIA):
                    col_metrics[i].metric(label, getattr(result, field_name))

                if tier == "Deep Dive":
                    st.success("Strong candidate — proceed to full research workflow.")
                elif tier == "Watchlist":
                    st.warning("Monitor — re-score when more data available.")
                else:
                    st.error("Pass — does not meet minimum threshold.")
            except ValueError as e:
                st.error(str(e))

    existing: List[GreenblattScore] = st.session_state[_SS_SCORES]
    if existing:
        st.divider()
        st.markdown("#### All Scored Spinoffs This Session")
        rows = []
        for s in existing:
            rows.append({
                "Ticker": s.ticker,
                "Company": s.company_name,
                "Total": s.total,
                "Tier": s.tier,
                **{label: getattr(s, fname) for fname, label, _ in CRITERIA},
                "Notes": s.notes,
            })
        df = pd.DataFrame(rows).sort_values("Total", ascending=False)
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "⬇️ Export Scores (JSON)",
            data=json.dumps([
                {
                    "ticker": s.ticker, "company": s.company_name,
                    "total": s.total, "tier": s.tier, "notes": s.notes,
                    **{fn: getattr(s, fn) for fn, _, _ in CRITERIA}
                }
                for s in existing
            ], indent=2),
            file_name="spinoff_greenblatt_scores.json",
            mime="application/json",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Management Promises
# ══════════════════════════════════════════════════════════════════════════════
def _tab_promises() -> None:
    st.subheader("🤝 Management Promises Tracker")

    promises: List[ManagementPromise] = st.session_state[_SS_PROMISES]

    if promises:
        rows = [
            {
                "Ticker":  p.ticker,
                "Date":    p.promise_date,
                "Promise": p.promise,
                "Source":  p.source,
                "Status":  p.status,
                "Notes":   p.notes,
            }
            for p in promises
        ]
        df = pd.DataFrame(rows)

        status_filter = st.multiselect(
            "Filter by Status", ["Pending", "Kept", "Broken", "Partially Met"],
            default=["Pending", "Kept", "Broken", "Partially Met"],
            key="promise_status_filter",
        )
        filtered = df[df["Status"].isin(status_filter)] if status_filter else df
        st.dataframe(filtered, use_container_width=True)

        kept     = sum(1 for p in promises if p.status == "Kept")
        broken   = sum(1 for p in promises if p.status == "Broken")
        pending  = sum(1 for p in promises if p.status == "Pending")
        partial  = sum(1 for p in promises if p.status == "Partially Met")
        total    = len(promises)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kept",         kept,    f"{kept/total:.0%}" if total else "—")
        c2.metric("Broken",       broken,  f"{broken/total:.0%}" if total else "—")
        c3.metric("Partially Met",partial, f"{partial/total:.0%}" if total else "—")
        c4.metric("Pending",      pending, f"{pending/total:.0%}" if total else "—")
    else:
        st.info("No promises tracked yet. Add one below.")

    st.divider()
    st.markdown("#### Log a New Promise")
    with st.form("promise_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        p_ticker  = c1.text_input("Ticker *", placeholder="e.g. GEV")
        p_source  = c2.selectbox("Source", ["10-K", "10-Q", "earnings call",
                                             "investor day", "press release", "other"])
        p_date    = st.date_input("Promise Date", value=date.today(), key="p_date")
        p_promise = st.text_area("Promise *", height=80,
                                 placeholder="e.g. Achieve $2B FCF by 2028")
        p_status  = st.selectbox("Status", ["Pending", "Kept", "Broken", "Partially Met"])
        p_notes   = st.text_area("Notes", height=60)
        submitted = st.form_submit_button("Log Promise", type="primary")

    if submitted:
        if not p_ticker or not p_promise:
            st.error("Ticker and Promise are required.")
        else:
            entry = ManagementPromise(
                ticker=p_ticker.upper(),
                promise_date=p_date.isoformat(),
                promise=p_promise,
                source=p_source,
                status=p_status,
                notes=p_notes,
            )
            st.session_state[_SS_PROMISES].append(entry)
            try:
                add_promise(entry, _PROMISES_CSV)
            except Exception:
                pass
            st.success(f"✅ Promise logged for {p_ticker.upper()}.")
            st.rerun()

    if promises:
        st.download_button(
            "⬇️ Export Promises (CSV)",
            data=pd.DataFrame([
                {"ticker": p.ticker, "promise_date": p.promise_date,
                 "promise": p.promise, "source": p.source,
                 "status": p.status, "notes": p.notes}
                for p in promises
            ]).to_csv(index=False),
            file_name="management_promises.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Thesis Tracker
# ══════════════════════════════════════════════════════════════════════════════
def _tab_thesis() -> None:
    st.subheader("📓 Investment Thesis Tracker")

    theses: List[ThesisEntry] = st.session_state[_SS_THESES]

    if theses:
        rows = [
            {
                "Ticker":        t.ticker,
                "Entry Date":    t.entry_date,
                "Target ($)":    t.target_price,
                "Catalyst":      t.catalyst,
                "Status":        t.current_thesis,
                "Hypothesis":    t.hypothesis,
                "Notes":         t.notes,
            }
            for t in theses
        ]
        df = pd.DataFrame(rows)
        status_filter = st.multiselect(
            "Filter by Status", ["Active", "Broken", "Realized"],
            default=["Active", "Broken", "Realized"], key="thesis_status_filter"
        )
        filtered = df[df["Status"].isin(status_filter)] if status_filter else df
        st.dataframe(filtered, use_container_width=True)

        active   = sum(1 for t in theses if t.current_thesis == "Active")
        realized = sum(1 for t in theses if t.current_thesis == "Realized")
        broken   = sum(1 for t in theses if t.current_thesis == "Broken")
        c1, c2, c3 = st.columns(3)
        c1.metric("Active",   active)
        c2.metric("Realized", realized)
        c3.metric("Broken",   broken)
    else:
        st.info("No theses tracked yet. Add one below.")

    st.divider()
    st.markdown("#### Add New Thesis")
    with st.form("thesis_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        t_ticker  = c1.text_input("Ticker *", placeholder="e.g. GEV")
        t_status  = c2.selectbox("Status", ["Active", "Broken", "Realized"])
        t_date    = st.date_input("Entry Date", value=date.today(), key="t_date")
        t_target  = st.number_input("Target Price ($)", min_value=0.0,
                                    step=0.50, format="%.2f")
        t_hyp     = st.text_area("Hypothesis *", height=100,
                                 placeholder="One paragraph on why this spinoff is mispriced.")
        t_cat     = st.text_input("Primary Catalyst",
                                  placeholder="e.g. TSA expiry + margin normalization")
        t_notes   = st.text_area("Notes", height=60)
        submitted = st.form_submit_button("Add Thesis", type="primary")

    if submitted:
        if not t_ticker or not t_hyp:
            st.error("Ticker and Hypothesis are required.")
        else:
            entry = ThesisEntry(
                ticker=t_ticker.upper(),
                hypothesis=t_hyp,
                entry_date=t_date.isoformat(),
                target_price=float(t_target),
                catalyst=t_cat,
                current_thesis=t_status,
                notes=t_notes,
            )
            st.session_state[_SS_THESES].append(entry)
            try:
                add_thesis(entry, _THESIS_CSV)
            except Exception:
                pass
            st.success(f"✅ Thesis added for {t_ticker.upper()}.")
            st.rerun()

    if theses:
        st.download_button(
            "⬇️ Export Theses (CSV)",
            data=pd.DataFrame([
                {"ticker": t.ticker, "hypothesis": t.hypothesis,
                 "entry_date": t.entry_date, "target_price": t.target_price,
                 "catalyst": t.catalyst, "current_thesis": t.current_thesis,
                 "notes": t.notes}
                for t in theses
            ]).to_csv(index=False),
            file_name="thesis_tracker.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Cost Log
# ══════════════════════════════════════════════════════════════════════════════
def _tab_cost_log() -> None:
    st.subheader("💰 Research Cost Log")
    st.caption("Track API and data costs per spinoff position.")

    # ── LLM API spend from core.cost ─────────────────────────────────────────
    with st.expander("🤖 LLM API Spend Today (core.cost)", expanded=True):
        spent = daily_spend()
        remaining = max(0.0, DAILY_CAP_USD - spent)
        pct = min(1.0, spent / DAILY_CAP_USD) if DAILY_CAP_USD > 0 else 0.0
        col1, col2, col3 = st.columns(3)
        col1.metric("Spent Today", f"${spent:.4f}")
        col2.metric("Daily Cap", f"${DAILY_CAP_USD:.2f}")
        col3.metric("Remaining", f"${remaining:.4f}")
        st.progress(pct, text=f"{pct*100:.1f}% of daily cap used")
        llm_entries = read_log()
        today_entries = [e for e in llm_entries if e.get("date") == date.today().isoformat()]
        if today_entries:
            st.dataframe(
                pd.DataFrame(today_entries)[
                    ["timestamp", "model", "technique", "caller",
                     "prompt_tokens", "completion_tokens", "cost_usd"]
                ],
                use_container_width=True,
            )
        else:
            st.info("No LLM calls logged today.")
    st.divider()

    costs: List[CostEntry] = st.session_state[_SS_COSTS]
    totals = total_by_category(costs)
    grand  = sum(totals.values())

    cols = st.columns(len(CATEGORIES) + 1)
    for i, cat in enumerate(CATEGORIES):
        cols[i].metric(cat, f"${totals[cat]:.2f}")
    cols[-1].metric("Total", f"${grand:.2f}")

    if costs:
        st.divider()
        rows = [
            {"Date": c.date, "Ticker": c.ticker, "Category": c.category,
             "Description": c.description, "Amount ($)": c.amount_usd}
            for c in costs
        ]
        df = pd.DataFrame(rows).sort_values("Date", ascending=False)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No costs logged yet.")

    st.divider()
    st.markdown("#### Log a Cost")
    with st.form("cost_form", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        c_date    = c1.date_input("Date", value=date.today(), key="c_date")
        c_cat     = c2.selectbox("Category", CATEGORIES)
        c_ticker  = c3.text_input("Ticker (optional)", placeholder="e.g. GEV")
        c_desc    = st.text_input("Description *", placeholder="e.g. 10-K analysis via GPT-4o")
        c_amount  = st.number_input("Amount ($)", min_value=0.0,
                                    step=0.01, format="%.4f")
        submitted = st.form_submit_button("Log Cost", type="primary")

    if submitted:
        if not c_desc:
            st.error("Description is required.")
        else:
            entry = CostEntry(
                date=c_date.isoformat(),
                category=c_cat,
                description=c_desc,
                amount_usd=float(c_amount),
                ticker=c_ticker.upper() if c_ticker else "",
            )
            st.session_state[_SS_COSTS].append(entry)
            try:
                log_cost(entry, _COSTS_CSV)
            except Exception:
                pass
            st.success(f"✅ Cost of ${c_amount:.4f} logged.")
            st.rerun()

    if costs:
        st.download_button(
            "⬇️ Export Cost Log (CSV)",
            data=pd.DataFrame([
                {"date": c.date, "category": c.category, "description": c.description,
                 "amount_usd": c.amount_usd, "ticker": c.ticker}
                for c in costs
            ]).to_csv(index=False),
            file_name="spinoff_cost_log.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# SPINOFF COMMITTEE — spinoff-specific agents + inline runner
# ══════════════════════════════════════════════════════════════════════════════

_SPINOFF_AGENTS = [
    {
        "role": "Spinoff Opportunity Analyst",
        "icon": "🔍",
        "system": (
            "You are a special situations analyst specializing in corporate spinoffs following "
            "Joel Greenblatt's framework. You identify forced selling dynamics, Wall Street neglect, "
            "hidden value, and insider alignment. Cite specific numbers from the evidence provided. "
            "Be direct and keep responses under 200 words."
        ),
    },
    {
        "role": "Value & Quality Analyst",
        "icon": "💰",
        "system": (
            "You are a value investor in the Graham/Greenblatt tradition assessing standalone business quality. "
            "Focus on returns on capital, FCF generation, balance sheet strength, and EV/EBIT vs peers. "
            "Demand numbers before accepting qualitative claims. "
            "Be direct, data-grounded, and keep responses under 200 words."
        ),
    },
    {
        "role": "Devil's Advocate",
        "icon": "⚠️",
        "system": (
            "Your job is to steelman the bear case on this spinoff. "
            "Surface what bulls are ignoring: hidden liabilities, management credibility gaps, "
            "balance sheet traps, or whether the parent dumped a bad business on shareholders. "
            "Be specific, challenging, and keep responses under 200 words."
        ),
    },
]

_SP_ROUND1 = """Spinoff Investment Proposal:
{proposal}

Provide your initial position:
1. Your stance given your role (1 sentence)
2. Top 2 signals that support this investment
3. Top 2 risks or gaps you see
4. Preliminary vote: APPROVE, REJECT, or MODIFY

Stay under 200 words. Cite specific numbers where available."""

_SP_ROUND2 = """Spinoff Investment Proposal:
{proposal}

Other committee members' initial positions:
{context}

Respond to the discussion:
1. Challenge the weakest argument from another member (name them explicitly)
2. Acknowledge one valid point from another member (name them)
3. State whether your position has shifted and why

Stay under 200 words. Be direct."""

_SP_ROUND3 = """Spinoff Investment Proposal:
{proposal}

Full committee discussion:
{context}

Cast your final vote. Use exactly this format:

VOTE: [APPROVE / REJECT / MODIFY]
JUSTIFICATION: [2–3 sentences citing specific evidence from the proposal or debate]"""


def _run_spinoff_committee(proposal: str, on_message=None) -> dict:
    """3-round spinoff committee debate with purpose-built agents."""
    import openai
    from core.investment_committee import MessageBus, Message
    from datetime import datetime

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    bus = MessageBus()
    agent_outputs = []

    def _call(system, user):
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.7,
            max_tokens=400,
        )
        return resp.choices[0].message.content.strip()

    def _parse_vote(text):
        u = text.upper()
        if "APPROVE" in u: return "APPROVE"
        if "REJECT" in u: return "REJECT"
        if "MODIFY" in u: return "MODIFY"
        return "ABSTAIN"

    # Round 1 — Initial positions
    for agent in _SPINOFF_AGENTS:
        content = _call(agent["system"], _SP_ROUND1.format(proposal=proposal))
        bus.send(Message(sender=agent["role"], recipient="all", content=content,
                         timestamp=datetime.now().isoformat(), message_type="proposal", round_num=1))
        agent_outputs.append({"round": 1, "agent": agent["role"], "icon": agent["icon"], "content": content})
        if on_message:
            on_message(1, agent["role"], agent["icon"], "proposal", content)

    # Round 2 — Challenges
    for agent in _SPINOFF_AGENTS:
        context = bus.context_for(agent["role"], before_round=2)
        content = _call(agent["system"], _SP_ROUND2.format(proposal=proposal, context=context))
        bus.send(Message(sender=agent["role"], recipient="all", content=content,
                         timestamp=datetime.now().isoformat(), message_type="challenge", round_num=2))
        agent_outputs.append({"round": 2, "agent": agent["role"], "icon": agent["icon"], "content": content})
        if on_message:
            on_message(2, agent["role"], agent["icon"], "challenge", content)

    # Round 3 — Final vote
    votes = {}
    for agent in _SPINOFF_AGENTS:
        content = _call(agent["system"], _SP_ROUND3.format(proposal=proposal, context=bus.full_transcript()))
        vote = _parse_vote(content)
        votes[agent["role"]] = {"vote": vote, "justification": content, "icon": agent["icon"]}
        bus.send(Message(sender=agent["role"], recipient="all", content=content,
                         timestamp=datetime.now().isoformat(), message_type="vote", round_num=3))
        agent_outputs.append({"round": 3, "agent": agent["role"], "icon": agent["icon"], "content": content, "vote": vote})
        if on_message:
            on_message(3, agent["role"], agent["icon"], "vote", content)

    # Tally
    tally: dict = {}
    for v in votes.values():
        tally[v["vote"]] = tally.get(v["vote"], 0) + 1
    n = len(_SPINOFF_AGENTS)
    if tally.get("APPROVE", 0) > n / 2:
        outcome = "APPROVED"
    elif tally.get("REJECT", 0) > n / 2:
        outcome = "REJECTED"
    elif tally.get("MODIFY", 0) > n / 2:
        outcome = "MODIFICATION REQUIRED"
    else:
        outcome = "NO CONSENSUS — REQUIRES FURTHER DISCUSSION"

    return {
        "proposal": proposal,
        "bus": bus,
        "votes": votes,
        "tally": tally,
        "outcome": outcome,
        "agent_outputs": agent_outputs,
        "message_count": len(bus.messages),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Research Chat
# ══════════════════════════════════════════════════════════════════════════════
def _tab_research_chat() -> None:
    st.subheader("💬 Research Chat")
    st.caption("Ask questions about the indexed company — answers grounded in SEC filings and research.")

    rag = st.session_state.get("rag_system")
    rag_count = rag.count() if rag else 0
    xbrl_tickers = list(st.session_state.get("xbrl_by_ticker", {}).keys())

    if rag_count > 0 or xbrl_tickers:
        ticker_str = ", ".join(xbrl_tickers) if xbrl_tickers else "indexed docs"
        st.success(f"✅ {rag_count} chunks indexed · **{ticker_str}** — chat is grounded in your research data.")
    else:
        st.warning(
            "No filings indexed yet. Go to **Layer 3 — Document RAG**, fetch the LION 10-K from EDGAR, "
            "then return here to chat."
        )
        return

    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("Enter your OpenAI API key in the sidebar.")
        return

    if "spinoff_chat_history" not in st.session_state:
        st.session_state.spinoff_chat_history = []

    _, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("🗑️ Clear", key="clear_spinoff_chat"):
            st.session_state.spinoff_chat_history = []
            st.rerun()

    # Suggested questions when chat is empty
    if not st.session_state.spinoff_chat_history:
        st.markdown("**Try asking:**")
        suggestions = [
            "What is the business model and how does it make money?",
            "Why was this company spun off from its parent?",
            "What are the key risk factors post-spinoff?",
            "What does management say about capital allocation?",
            "What is the leverage and debt maturity profile?",
            "What insider ownership or equity grants exist?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(suggestions):
            if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state.spinoff_chat_history.append({"role": "user", "content": q})
                st.rerun()

    # Render existing messages
    for msg in st.session_state.spinoff_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 {len(msg['sources'])} source(s)", expanded=False):
                    for src in msg["sources"]:
                        st.caption(f"**{src.source}** · relevance {src.relevance_score:.2f}")
                        st.caption(src.content[:200] + ("..." if len(src.content) > 200 else ""))

    # Chat input
    question = st.chat_input("Ask about this spinoff...")
    if question:
        st.session_state.spinoff_chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching filings..."):
                try:
                    response = rag.answer_question(question, k=10)
                    st.markdown(response.answer)
                    st.caption(f"Confidence: {response.confidence}")
                    if response.sources:
                        with st.expander(f"📎 {len(response.sources)} source(s)", expanded=False):
                            for src in response.sources:
                                st.caption(f"**{src.source}** · relevance {src.relevance_score:.2f}")
                                st.caption(src.content[:200] + ("..." if len(src.content) > 200 else ""))
                    st.session_state.spinoff_chat_history.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                    })
                except Exception as e:
                    err = f"Error: {e}"
                    st.error(err)
                    st.session_state.spinoff_chat_history.append({"role": "assistant", "content": err})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — IC Debate
# ══════════════════════════════════════════════════════════════════════════════
def _tab_ic_debate() -> None:
    st.subheader("🏛️ Investment Committee Debate")
    st.caption(
        "Three spinoff specialists debate your thesis using evidence from indexed SEC filings.  \n"
        "**🔍 Spinoff Opportunity Analyst** · **💰 Value & Quality Analyst** · **⚠️ Devil's Advocate**"
    )

    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("Enter your OpenAI API key in the sidebar.")
        return

    theses: List[ThesisEntry] = st.session_state.get(_SS_THESES, [])
    watchlist: pd.DataFrame = st.session_state.get(_SS_WATCHLIST, pd.DataFrame())

    tickers_from_theses = [t.ticker for t in theses]
    wl_col = "ticker" if not watchlist.empty and "ticker" in watchlist.columns else None
    tickers_from_watchlist = list(watchlist[wl_col].dropna().str.upper()) if wl_col else []
    all_tickers = sorted(set(tickers_from_theses + tickers_from_watchlist) - {"LION"})

    col_a, col_b = st.columns([1, 2])
    with col_a:
        selected_ticker = st.selectbox(
            "Spinoff ticker",
            ["LION"] + all_tickers,
            key="ic_ticker_select",
        )

    thesis = next((t for t in theses if t.ticker == selected_ticker), None)
    with col_b:
        if thesis:
            excerpt = thesis.hypothesis[:160] + "..." if len(thesis.hypothesis) > 160 else thesis.hypothesis
            st.info(f"**{selected_ticker}** — {excerpt}")
        else:
            st.caption(f"No saved thesis for {selected_ticker}. Enter your proposal below.")

    # RAG status
    rag = st.session_state.get("rag_system")
    rag_count = rag.count() if rag else 0
    xbrl_by_ticker = st.session_state.get("xbrl_by_ticker", {})
    indexed_tickers = list(xbrl_by_ticker.keys())

    if rag_count > 0:
        st.success(
            f"✅ {rag_count} chunks indexed"
            + (f" · **{', '.join(indexed_tickers)}**" if indexed_tickers else "")
            + " — committee will cite filing evidence in their arguments."
        )
    else:
        st.info("💡 No filings indexed. Go to Layer 3 to index LION filings for a grounded debate. Committee can still debate from the thesis text.")

    st.divider()

    # Auto-fill proposal from thesis tracker
    default_proposal = ""
    if thesis:
        default_proposal = (
            f"Spinoff: {selected_ticker}\n\n"
            f"Thesis: {thesis.hypothesis}\n\n"
            f"Primary Catalyst: {thesis.catalyst}\n"
            f"Target Price: ${thesis.target_price:.2f}"
        )

    proposal_text = st.text_area(
        "Investment proposal for the committee",
        value=default_proposal,
        height=160,
        key="ic_debate_proposal",
        placeholder=(
            "Describe the LION spinoff thesis — e.g. spun off from [Parent], trades at [X]x EV/EBIT, "
            "[Y] index funds forced to sell, management holds [Z]% equity..."
        ),
    )

    run_debate = st.button(
        "🏛️ Convene Committee",
        type="primary",
        disabled=not proposal_text.strip(),
        key="run_spinoff_debate",
    )

    if run_debate and proposal_text.strip():
        st.session_state.pop("sl_ic_result", None)

        # Enrich proposal with top RAG hits
        enriched = proposal_text.strip()
        if rag and rag_count > 0:
            try:
                hits = rag.search(proposal_text[:300], k=5)
                if hits:
                    snippets = "\n\n".join(f"[{r.source}]: {r.content[:300]}" for r in hits)
                    enriched += f"\n\nEvidence from SEC filings:\n{snippets}"
            except Exception:
                pass

        _ROUND_LABELS = {1: "📋 Initial Position", 2: "⚔️ Challenge", 3: "🗳️ Final Vote"}

        with st.status("🏛️ Spinoff Committee in session...", expanded=True) as _status:
            _status.write("Convening Spinoff Opportunity Analyst, Value & Quality Analyst, and Devil's Advocate...")

            def _on_msg(round_num, agent_role, icon, msg_type, content):
                _status.write(f"{icon} **{agent_role}** — {_ROUND_LABELS[round_num]}")

            try:
                result = _run_spinoff_committee(enriched, on_message=_on_msg)
                st.session_state["sl_ic_result"] = result
                _status.update(label="✅ Committee deliberation complete", state="complete", expanded=False)
            except Exception as e:
                _status.update(label="❌ Error", state="error")
                st.error(f"Error: {e}")

    # ── Results ────────────────────────────────────────────────────────────────
    if "sl_ic_result" in st.session_state:
        result = st.session_state["sl_ic_result"]
        outcome = result["outcome"]
        tally = result["tally"]

        st.divider()

        if outcome == "APPROVED":
            st.success(f"## ✅ {outcome}")
        elif outcome == "REJECTED":
            st.error(f"## ❌ {outcome}")
        elif outcome == "MODIFICATION REQUIRED":
            st.warning(f"## 🔧 {outcome}")
        else:
            st.warning(f"## ⚠️ {outcome}")

        tally_cols = st.columns(len(tally) + 1)
        tally_cols[0].metric("Messages exchanged", result["message_count"])
        for i, (vote_type, count) in enumerate(tally.items(), 1):
            tally_cols[i].metric(vote_type, f"{count} / {len(result['votes'])}")

        st.divider()

        tab_r1, tab_r2, tab_r3 = st.tabs(["📋 Round 1 — Positions", "⚔️ Round 2 — Challenges", "🗳️ Round 3 — Votes"])

        def _render_round(tab, round_num):
            with tab:
                messages = result["bus"].get_by_round(round_num)
                for msg in messages:
                    icon = next(
                        (a["icon"] for a in result["agent_outputs"]
                         if a["agent"] == msg.sender and a["round"] == round_num),
                        "🤖",
                    )
                    with st.expander(f"{icon} {msg.sender}", expanded=True):
                        if round_num == 3:
                            vote = result["votes"].get(msg.sender, {}).get("vote", "")
                            badge = {"APPROVE": "✅", "REJECT": "❌", "MODIFY": "🔧"}.get(vote, "")
                            st.markdown(f"**Vote: {badge} {vote}**")
                        st.markdown(msg.content)

        _render_round(tab_r1, 1)
        _render_round(tab_r2, 2)
        _render_round(tab_r3, 3)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def render() -> None:
    _init_state()

    st.title("🔬 Special Situations Lab — Spinoff Research")
    st.caption(
        "A Greenblatt-style framework for researching corporate spinoffs. "
        "Use Layer 3 (Document RAG) to index SEC filings; analyze them here."
    )

    with st.expander("ℹ️ About This Lab", expanded=False):
        st.markdown("""
**Why spinoffs?**
Joel Greenblatt documented in *You Can Be a Stock Market Genius* that spinoffs
systematically outperform because:

1. **Forced selling** — index funds and parent shareholders dump shares immediately.
2. **Neglect** — Wall Street coverage is thin in the first 6–18 months.
3. **Hidden value** — parents bury their best (or worst) assets inside larger businesses.
4. **Insider alignment** — management receives equity in the spinco, not the parent.

**Workflow**
1. **Spinoff Setup** — add candidates to your watchlist.
2. **SEC Documents** — use Layer 3 to index Form 10 / 10-K filings.
3. **Greenblatt Screen** — score each candidate on 7 criteria (0–5 each, max 35).
4. **Management Promises** — log and track commitments made at spin-off.
5. **Thesis Tracker** — document your investment hypothesis and monitor it.
6. **Cost Log** — track API and research costs per position.
7. **Research Chat** — ask questions about the indexed company, grounded in SEC filings.
8. **IC Debate** — three spinoff specialists debate your thesis using filing evidence.
""")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📋 Spinoff Setup",
        "📄 SEC Documents",
        "🎯 Greenblatt Screen",
        "🤝 Management Promises",
        "📓 Thesis Tracker",
        "💰 Cost Log",
        "💬 Research Chat",
        "🏛️ IC Debate",
    ])

    with tab1:
        _tab_setup()
    with tab2:
        _tab_sec_documents()
    with tab3:
        _tab_greenblatt()
    with tab4:
        _tab_promises()
    with tab5:
        _tab_thesis()
    with tab6:
        _tab_cost_log()
    with tab7:
        _tab_research_chat()
    with tab8:
        _tab_ic_debate()

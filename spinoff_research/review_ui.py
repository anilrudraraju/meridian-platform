"""
Phase 7 — Review UI for extracted spin-off field values.

Standalone Streamlit module; rendered from app.py the same way
special_situations/spinoff_lab.py is:
    if st.session_state.active_layer == "spinoff_research":
        review_ui.render()

Shows, per transaction: every extracted field with its value, status,
confidence, and source citations, plus approve/edit/reject actions that
write through review.py — never mutates the original extraction, per
schema.sql's audit-trail design.
"""
import streamlit as st

from spinoff_research.db import init_db
from spinoff_research.repository import list_transactions
from spinoff_research.review_data import load_review_rows, missing_deterministic_fields
from spinoff_research.review import record_review
from spinoff_research.status import ReviewAction

_STATUS_BADGE = {
    "extracted_high_confidence": "🟢",
    "extracted_uncertain": "🟡",
    "not_found": "⚪",
    "conflicting_sources": "🔴",
    "not_yet_determinable": "⚪",
    "requires_manual_review": "🟠",
    "approved": "✅",
    "rejected": "❌",
    "manually_entered": "✏️",
}


def _get_conn():
    """
    Deliberately NOT @st.cache_resource: a cached sqlite3.Connection
    survives across Streamlit script reruns, but Streamlit's script-runner
    can execute a rerun (e.g. triggered by st.rerun() inside a button
    callback) on a different thread than the one that created the
    connection — confirmed live: 'SQLite objects created in a thread can
    only be used in that same thread.' SQLite connections to a local file
    are cheap to open; a fresh connection per render avoids the whole
    class of cross-thread reuse bugs.
    """
    return init_db()


def _status_badge(status: str) -> str:
    return f"{_STATUS_BADGE.get(status, '⚪')} {status}"


def _confidence_str(confidence) -> str:
    return f"{confidence:.0%}" if confidence is not None else "—"


def render() -> None:
    conn = _get_conn()

    st.title("📋 Spin-off Field Review")
    st.caption(
        "Extracted field values across ingested transactions — confirm, correct, or reject each one. "
        "Every value keeps a citation back to its source; corrections never overwrite the original extraction."
    )

    with st.expander("ℹ️ About This View", expanded=False):
        st.markdown("""
This is Phase 7 of the spin-off research prototype: a review surface over what Phases 2-5 built —
field definitions, SEC/XBRL/Form 4 extraction, and persisted results with provenance.

**Currently covers 19 deterministic fields** (SEC filing metadata, XBRL, Form 4 insider-buying
aggregation, yFinance). AI-assisted fields (21 more, e.g. "did the CEO come from the parent?")
are not built yet — that's the next phase.

**Status colors:** 🟢 high confidence · 🟡 uncertain · 🔴 conflicting sources · ⚪ not found ·
✅ approved · ❌ rejected · ✏️ manually corrected
""")

    transactions = list_transactions(conn)
    if not transactions:
        st.info(
            "No transactions ingested yet. Run the extraction orchestrator "
            "(spinoff_research.orchestrator.run_deterministic_extraction) against a transaction first."
        )
        return

    labels = [t.label() for t in transactions]
    selected_idx = st.selectbox(
        "Transaction", options=range(len(transactions)),
        format_func=lambda i: labels[i], key="review_txn_select",
    )
    transaction = transactions[selected_idx]

    rows = load_review_rows(conn, transaction.transaction_id)
    if not rows:
        st.warning("No fields extracted for this transaction yet.")
        return

    # ── Summary metrics ──────────────────────────────────────────────────
    total = len(rows)
    high_conf = sum(1 for r in rows if r.status == "extracted_high_confidence")
    uncertain = sum(1 for r in rows if r.status == "extracted_uncertain")
    not_found = sum(1 for r in rows if r.status == "not_found")
    reviewed = sum(1 for r in rows if r.status in ("approved", "rejected", "manually_entered"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Fields extracted", total)
    c2.metric("High confidence", high_conf)
    c3.metric("Needs review", uncertain)
    c4.metric("Not found", not_found)
    c5.metric("Reviewed", reviewed)

    missing = missing_deterministic_fields(rows)
    if missing:
        st.caption(f"⚠️ {len(missing)} deterministic field(s) not yet attempted: {', '.join(missing)}")

    st.divider()

    # ── Summary table ─────────────────────────────────────────────────────
    st.markdown("#### All Fields")
    import pandas as pd
    table_data = [
        {
            "Field": r.display_name,
            "Category": r.category.replace("_", " ").title(),
            "Value": r.raw_value or "—",
            "Status": _status_badge(r.status),
            "Confidence": _confidence_str(r.confidence),
            "Sources": len(r.sources),
        }
        for r in rows
    ]
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    st.divider()

    # ── Per-field review ────────────────────────────────────────────────────
    st.markdown("#### Review Individual Fields")
    status_filter = st.multiselect(
        "Filter by status",
        options=sorted({r.status for r in rows}),
        default=[s for s in ("extracted_uncertain", "conflicting_sources", "not_found") if s in {r.status for r in rows}],
        key="review_status_filter",
    )
    filtered_rows = [r for r in rows if not status_filter or r.status in status_filter]

    for row in filtered_rows:
        with st.expander(f"{_status_badge(row.status)} **{row.display_name}** — {row.raw_value or 'no value'}"):
            col_info, col_action = st.columns([2, 1])

            with col_info:
                st.caption(f"Field key: `{row.field_key}` · Extraction: {row.extraction_category}")
                st.write(f"**Normalized value:** {row.raw_value or '—'}")
                if row.confidence is not None:
                    st.write(f"**Confidence:** {row.confidence:.0%}")
                if row.as_of_date:
                    st.write(f"**As of:** {row.as_of_date}")

                if row.sources:
                    st.markdown("**Sources:**")
                    for s in row.sources:
                        if s.supporting_excerpt:
                            st.code(s.supporting_excerpt, language=None)
                        if s.reasoning_summary:
                            st.caption(s.reasoning_summary)
                        if s.sec_url:
                            st.markdown(f"[View filing]({s.sec_url}) · {s.filing_type or ''}")
                else:
                    st.caption("No source citation recorded.")

            with col_action:
                already_reviewed = row.status in ("approved", "rejected", "manually_entered")
                if already_reviewed:
                    st.info(f"Already {row.status}.")
                else:
                    if st.button("✅ Approve", key=f"approve_{row.field_value_id}", use_container_width=True):
                        record_review(conn, row.field_value_id, ReviewAction.APPROVE, reviewed_by="Rich")
                        st.rerun()
                    if st.button("❌ Reject", key=f"reject_{row.field_value_id}", use_container_width=True):
                        record_review(conn, row.field_value_id, ReviewAction.REJECT, reviewed_by="Rich")
                        st.rerun()

                    with st.form(f"edit_form_{row.field_value_id}"):
                        corrected = st.text_input("Correct value", key=f"correct_{row.field_value_id}")
                        notes = st.text_area("Notes", key=f"notes_{row.field_value_id}", height=60)
                        if st.form_submit_button("✏️ Save Correction", use_container_width=True):
                            if corrected:
                                record_review(
                                    conn, row.field_value_id, ReviewAction.EDIT,
                                    corrected_value=corrected, reviewer_notes=notes, reviewed_by="Rich",
                                )
                                st.rerun()
                            else:
                                st.error("Enter a corrected value first.")

                    if st.button("🚫 Mark Not Found", key=f"notfound_{row.field_value_id}", use_container_width=True):
                        record_review(conn, row.field_value_id, ReviewAction.MARK_NOT_FOUND, reviewed_by="Rich")
                        st.rerun()

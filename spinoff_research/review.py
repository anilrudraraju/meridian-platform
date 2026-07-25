"""
Human review actions on extracted field_values.

Reviewing a value never mutates the original field_values row — per
schema.sql's own design (is_original_extraction, superseded_by_field_value_id)
and the brief's requirement that a corrected value must preserve the
original. record_review() implements that: 'edit' and 'mark_not_found'
insert a NEW field_values row (the correction) and point the OLD row at it
via superseded_by_field_value_id; 'approve' and 'reject' don't create a new
value at all, they just log the decision against the existing row.
"""
import sqlite3
from typing import Optional

from spinoff_research.status import FieldStatus, ReviewAction


def record_review(
    conn: sqlite3.Connection,
    field_value_id: int,
    action: ReviewAction,
    corrected_value: Optional[str] = None,
    reviewer_notes: Optional[str] = None,
    reviewed_by: Optional[str] = None,
) -> int:
    """
    Returns the new review_id. For 'edit'/'mark_not_found', also creates a
    new field_values row (status='manually_entered' or 'not_found') and
    marks the original superseded — callers should re-query
    get_current_field_value() afterward rather than assume the row they
    had is still current.
    """
    original = conn.execute(
        "SELECT * FROM field_values WHERE field_value_id = ?", (field_value_id,)
    ).fetchone()
    if original is None:
        raise ValueError(f"No field_values row with id {field_value_id}")

    conn.execute(
        """INSERT INTO field_reviews (field_value_id, action, corrected_value, reviewer_notes, reviewed_by)
           VALUES (?, ?, ?, ?, ?)""",
        (field_value_id, action.value, corrected_value, reviewer_notes, reviewed_by),
    )
    review_id = conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]

    if action == ReviewAction.APPROVE:
        conn.execute(
            "UPDATE field_values SET status = ? WHERE field_value_id = ?",
            (FieldStatus.APPROVED.value, field_value_id),
        )

    elif action == ReviewAction.REJECT:
        conn.execute(
            "UPDATE field_values SET status = ? WHERE field_value_id = ?",
            (FieldStatus.REJECTED.value, field_value_id),
        )

    elif action == ReviewAction.EDIT:
        new_id = _insert_corrected_value(
            conn, original, corrected_value, FieldStatus.MANUALLY_ENTERED
        )
        conn.execute(
            "UPDATE field_values SET is_original_extraction = 0, superseded_by_field_value_id = ? "
            "WHERE field_value_id = ?",
            (new_id, field_value_id),
        )

    elif action == ReviewAction.MARK_NOT_FOUND:
        new_id = _insert_corrected_value(
            conn, original, None, FieldStatus.NOT_FOUND
        )
        conn.execute(
            "UPDATE field_values SET is_original_extraction = 0, superseded_by_field_value_id = ? "
            "WHERE field_value_id = ?",
            (new_id, field_value_id),
        )

    conn.commit()
    return review_id


def _insert_corrected_value(
    conn: sqlite3.Connection,
    original: sqlite3.Row,
    corrected_value: Optional[str],
    status: FieldStatus,
) -> int:
    cur = conn.execute(
        """INSERT INTO field_values
           (transaction_id, field_key, raw_value, normalized_value, unit, currency, scale,
            reporting_period, as_of_date, value_basis, confidence, status, is_original_extraction)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
        (original["transaction_id"], original["field_key"], corrected_value, corrected_value,
         original["unit"], original["currency"], original["scale"], original["reporting_period"],
         original["as_of_date"], "manually_corrected" if corrected_value else original["value_basis"],
         1.0 if corrected_value else None, status.value),
    )
    return cur.lastrowid


def get_review_history(conn: sqlite3.Connection, field_value_id: int):
    """All review rows for a field_value_id, oldest first — the audit trail."""
    return conn.execute(
        "SELECT * FROM field_reviews WHERE field_value_id = ? ORDER BY reviewed_at ASC",
        (field_value_id,),
    ).fetchall()

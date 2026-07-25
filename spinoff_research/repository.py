"""
Persistence layer for companies and spinoff_transactions.

models.py's Company/SpinoffTransaction are plain dataclasses with no
persistence of their own (by design — see their docstrings). This module
is the missing piece: given an in-memory dataclass, get or create the
matching row, and return the dataclass with its *_id populated. Document
ingestion (ingestion.py) requires a real transaction_id to attach documents
to, so this has to exist before ingestion can write anything.
"""
import sqlite3
from typing import Optional

from spinoff_research.models import Company, SpinoffTransaction


def get_or_create_company(conn: sqlite3.Connection, company: Company) -> Company:
    """
    Match on (name, ticker) — the same uniqueness constraint schema.sql
    enforces. Updates cik/sector/industry on an existing row if the
    incoming Company has values the stored row doesn't (never overwrites a
    stored non-null value with a new null, since resolution can happen in
    stages: CIK first via sec_service, sector/industry later via market data).
    """
    row = conn.execute(
        "SELECT * FROM companies WHERE name = ? AND (ticker = ? OR (ticker IS NULL AND ? IS NULL))",
        (company.name, company.ticker, company.ticker),
    ).fetchone()

    if row is None:
        cur = conn.execute(
            "INSERT INTO companies (name, ticker, cik, sector, industry) VALUES (?, ?, ?, ?, ?)",
            (company.name, company.ticker, company.cik, company.sector, company.industry),
        )
        conn.commit()
        company.company_id = cur.lastrowid
        return company

    updates = {}
    for field, incoming in (("cik", company.cik), ("sector", company.sector), ("industry", company.industry)):
        if incoming and not row[field]:
            updates[field] = incoming
    if updates:
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(
            f"UPDATE companies SET {set_clause} WHERE company_id = ?",
            (*updates.values(), row["company_id"]),
        )
        conn.commit()

    return Company(
        company_id=row["company_id"],
        name=row["name"],
        ticker=row["ticker"],
        cik=updates.get("cik", row["cik"]),
        sector=updates.get("sector", row["sector"]),
        industry=updates.get("industry", row["industry"]),
    )


def get_or_create_transaction(conn: sqlite3.Connection, transaction: SpinoffTransaction) -> SpinoffTransaction:
    """
    Persists parent and spinoff companies first (transactions FK-reference
    both), then the transaction itself, matched on the
    (parent_company_id, spinoff_company_id) uniqueness constraint.
    """
    parent = get_or_create_company(conn, transaction.parent)
    spinoff = get_or_create_company(conn, transaction.spinoff)

    row = conn.execute(
        "SELECT * FROM spinoff_transactions WHERE parent_company_id = ? AND spinoff_company_id = ?",
        (parent.company_id, spinoff.company_id),
    ).fetchone()

    if row is None:
        cur = conn.execute(
            "INSERT INTO spinoff_transactions "
            "(source_row_number, parent_company_id, spinoff_company_id, announcement_date, spinoff_date, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (transaction.source_row_number, parent.company_id, spinoff.company_id,
             transaction.announcement_date, transaction.spinoff_date, transaction.status),
        )
        conn.commit()
        transaction_id = cur.lastrowid
    else:
        transaction_id = row["transaction_id"]
        updates = {}
        if transaction.spinoff_date and not row["spinoff_date"]:
            updates["spinoff_date"] = transaction.spinoff_date
        if transaction.announcement_date and not row["announcement_date"]:
            updates["announcement_date"] = transaction.announcement_date
        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            conn.execute(
                f"UPDATE spinoff_transactions SET {set_clause} WHERE transaction_id = ?",
                (*updates.values(), transaction_id),
            )
            conn.commit()

    return SpinoffTransaction(
        transaction_id=transaction_id,
        parent=parent,
        spinoff=spinoff,
        announcement_date=transaction.announcement_date or (row["announcement_date"] if row else None),
        spinoff_date=transaction.spinoff_date or (row["spinoff_date"] if row else None),
        source_row_number=transaction.source_row_number,
        status=transaction.status,
    )


def get_transaction(conn: sqlite3.Connection, transaction_id: int) -> Optional[SpinoffTransaction]:
    row = conn.execute(
        "SELECT * FROM spinoff_transactions WHERE transaction_id = ?", (transaction_id,)
    ).fetchone()
    if row is None:
        return None
    parent_row = conn.execute(
        "SELECT * FROM companies WHERE company_id = ?", (row["parent_company_id"],)
    ).fetchone()
    spinoff_row = conn.execute(
        "SELECT * FROM companies WHERE company_id = ?", (row["spinoff_company_id"],)
    ).fetchone()
    return SpinoffTransaction(
        transaction_id=row["transaction_id"],
        parent=Company(**{k: parent_row[k] for k in ("company_id", "name", "ticker", "cik", "sector", "industry")}),
        spinoff=Company(**{k: spinoff_row[k] for k in ("company_id", "name", "ticker", "cik", "sector", "industry")}),
        announcement_date=row["announcement_date"],
        spinoff_date=row["spinoff_date"],
        source_row_number=row["source_row_number"],
        status=row["status"],
    )


def list_transactions(conn: sqlite3.Connection) -> list:
    """All persisted transactions, newest first — for a UI picker."""
    rows = conn.execute(
        "SELECT transaction_id FROM spinoff_transactions ORDER BY created_at DESC"
    ).fetchall()
    return [get_transaction(conn, r["transaction_id"]) for r in rows]

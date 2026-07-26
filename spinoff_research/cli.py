"""
Command-line entry point for the spin-off research pipeline. No UI —
input is a ticker list or a prior reviewed workbook, output is a
versioned Excel file. See design decisions in the conversation that
produced this: SQLite stays source of truth, Excel is a round-trip view.

Usage:
    # First run — extract a fresh batch of spin-offs.
    python -m spinoff_research.cli extract --tickers GE:GEV,SNY:INBX,ILMN:GRAL \\
        --announcement-dates 2021-11-09,2024-01-23,2023-12-17 \\
        --distribution-dates 2024-04-02,2024-05-30,2024-06-25 \\
        --out-dir ./exports

    # Clarification round — re-ingest Rich's markup, re-attempt only
    # rejected/needs-fix fields with no supplied correction, leave
    # approved and corrected fields untouched, write a new version.
    python -m spinoff_research.cli extract --resume ./exports/spinoff_review_v1_2026-07-26.xlsx \\
        --out-dir ./exports
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from spinoff_research.cost_tracking import read_log, total_cost_usd
from spinoff_research.db import init_db
from spinoff_research.models import Company, SpinoffTransaction
from spinoff_research.repository import get_or_create_transaction, get_transaction
from spinoff_research.orchestrator import run_deterministic_extraction
from spinoff_research.sec_service import resolve_cik
from spinoff_research.xlsx_writer import write_workbook, RunLogEntry
from spinoff_research.xlsx_reader import reingest_workbook


def _parse_ticker_pairs(raw: str) -> List[tuple]:
    """'GE:GEV,SNY:INBX' -> [('GE','GEV'), ('SNY','INBX')]"""
    pairs = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Expected 'PARENT:SPINOFF' format, got '{chunk}'")
        parent, spinoff = chunk.split(":", 1)
        pairs.append((parent.strip().upper(), spinoff.strip().upper()))
    return pairs


def _parse_date_list(raw: Optional[str], expected_len: int, label: str) -> List[Optional[str]]:
    if not raw:
        return [None] * expected_len
    dates = [d.strip() or None for d in raw.split(",")]
    if len(dates) != expected_len:
        raise ValueError(f"--{label} has {len(dates)} entries but {expected_len} tickers were given")
    return dates


def _next_version_path(out_dir: Path, base_name: str = "spinoff_review") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(f"{base_name}_v*.xlsx"))
    next_v = 1
    for p in existing:
        try:
            v = int(p.stem.split("_v")[-1].split("_")[0])
            next_v = max(next_v, v + 1)
        except ValueError:
            continue
    date_str = datetime.now().strftime("%Y-%m-%d")
    return out_dir / f"{base_name}_v{next_v}_{date_str}.xlsx"


def cmd_extract(args: argparse.Namespace) -> int:
    conn = init_db()
    out_dir = Path(args.out_dir)
    run_log: List[RunLogEntry] = []
    transaction_ids: List[int] = []
    run_started_at = datetime.now().isoformat()

    if args.resume:
        print(f"Re-ingesting reviewed workbook: {args.resume}")
        result = reingest_workbook(conn, Path(args.resume), reviewed_by=args.reviewed_by)
        print(f"  Approved: {result.approved}  Corrected: {result.corrected}  "
              f"Rejected (no fix): {result.rejected_no_fix}  Not yet reviewed: {result.skipped_not_reviewed}")
        if result.errors:
            print(f"  {len(result.errors)} issue(s) during re-ingestion:")
            for e in result.errors[:20]:
                print(f"    - {e}")

        # Re-run extraction per transaction present in the resumed
        # workbook, skipping any field_key currently approved or already
        # manually corrected — those are final per the "approved fields
        # are never touched again" rule.
        import openpyxl
        wb = openpyxl.load_workbook(args.resume, data_only=True)
        ws = wb["Input"]
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row and row[0]:
                transaction_ids.append(int(row[0]))
        transaction_ids = sorted(set(transaction_ids))

        for txn_id in transaction_ids:
            txn = get_transaction(conn, txn_id)
            if txn is None:
                continue
            locked = {
                r["field_key"] for r in conn.execute(
                    "SELECT field_key FROM field_values WHERE transaction_id = ? "
                    "AND status IN ('approved', 'manually_entered') "
                    "AND superseded_by_field_value_id IS NULL",
                    (txn_id,),
                ).fetchall()
            }
            summary = run_deterministic_extraction(conn, txn, skip_field_keys=locked)
            print(f"  {txn.label()}: {summary.field_count} field(s) re-attempted, "
                  f"{len(locked)} locked-in field(s) left untouched")
            for key in locked:
                run_log.append(RunLogEntry(txn.label(), key, "skipped_approved", "locked in from a prior review round"))
            for r in summary.results:
                run_log.append(RunLogEntry(txn.label(), r.field_key, "reattempted", f"status={r.status.value}"))

    else:
        pairs = _parse_ticker_pairs(args.tickers)
        ann_dates = _parse_date_list(args.announcement_dates, len(pairs), "announcement-dates")
        dist_dates = _parse_date_list(args.distribution_dates, len(pairs), "distribution-dates")

        if args.batch_size and len(pairs) > args.batch_size:
            print(f"Note: {len(pairs)} transactions requested, --batch-size {args.batch_size} — "
                  f"processing the first {args.batch_size} only. Re-run with the remainder separately.")
            pairs = pairs[:args.batch_size]
            ann_dates = ann_dates[:args.batch_size]
            dist_dates = dist_dates[:args.batch_size]

        for (parent_t, spinoff_t), ann_date, dist_date in zip(pairs, ann_dates, dist_dates):
            print(f"Resolving {parent_t} -> {spinoff_t} ...")
            parent = resolve_cik(parent_t)
            spinoff = resolve_cik(spinoff_t)
            if parent is None or spinoff is None:
                missing = parent_t if parent is None else spinoff_t
                print(f"  Could not resolve ticker '{missing}' — skipping this pair.")
                continue

            tx = SpinoffTransaction(
                parent=Company(name=parent["name"], ticker=parent_t, cik=parent["cik"]),
                spinoff=Company(name=spinoff["name"], ticker=spinoff_t, cik=spinoff["cik"]),
                announcement_date=ann_date, spinoff_date=dist_date,
            )
            saved = get_or_create_transaction(conn, tx)
            transaction_ids.append(saved.transaction_id)

            summary = run_deterministic_extraction(conn, saved)
            print(f"  {summary.field_count} field(s) extracted, {len(summary.errors)} skipped")
            for r in summary.results:
                run_log.append(RunLogEntry(saved.label(), r.field_key, "newly_extracted", f"status={r.status.value}"))
            for err in summary.errors:
                run_log.append(RunLogEntry(saved.label(), "-", "skipped_no_mechanism", err))

    if not transaction_ids:
        print("No transactions to export — nothing written.")
        return 1

    out_path = _next_version_path(out_dir)
    write_workbook(conn, transaction_ids, out_path, run_log=run_log)
    print(f"\nWorkbook written: {out_path}")

    this_run_calls = read_log(since=run_started_at)
    if this_run_calls:
        this_run_cost = sum(c["cost_usd"] for c in this_run_calls)
        print(f"\nAI cost — this run: ${this_run_cost:.4f} ({len(this_run_calls)} call(s))  |  "
              f"lifetime total: ${total_cost_usd():.4f}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spinoff_research", description="Spin-off research extraction CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract fields for a batch of spin-offs, or re-ingest a reviewed workbook")
    p_extract.add_argument("--tickers", help="Comma-separated PARENT:SPINOFF pairs, e.g. GE:GEV,SNY:INBX")
    p_extract.add_argument("--announcement-dates", help="Comma-separated ISO dates, aligned with --tickers order")
    p_extract.add_argument("--distribution-dates", help="Comma-separated ISO dates, aligned with --tickers order")
    p_extract.add_argument("--batch-size", type=int, default=None, help="Max transactions to process in this run")
    p_extract.add_argument("--resume", help="Path to a previously-exported, Rich-reviewed .xlsx to re-ingest and re-run")
    p_extract.add_argument("--reviewed-by", default="Rich", help="Name recorded against review actions from --resume")
    p_extract.add_argument("--out-dir", default="./exports", help="Directory for the versioned output workbook")
    p_extract.set_defaults(func=cmd_extract)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.resume and not args.tickers:
        parser.error("extract requires either --tickers (new batch) or --resume (clarification round)")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

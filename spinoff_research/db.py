"""
SQLite connection + schema bootstrap for the spin-off research prototype.

Deliberately separate from Meridian's ChromaDB/JSONL persistence
(CLAUDE.md's hardcoded /tmp/meridian_chromadb rule governs vector search,
not this store) — this data must survive process restarts, which /tmp
does not guarantee on Streamlit Cloud.
"""
import os
import sqlite3
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_DEFAULT_DB_PATH = Path(__file__).parent / "data" / "spinoff_research.db"


def get_db_path() -> Path:
    """DB path is env-configurable; defaults to a file alongside this package."""
    override = os.environ.get("SPINOFF_RESEARCH_DB_PATH")
    return Path(override) if override else _DEFAULT_DB_PATH


def get_connection(db_path: Path = None) -> sqlite3.Connection:
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path = None) -> sqlite3.Connection:
    """Create all tables if they don't exist. Idempotent."""
    conn = get_connection(db_path)
    schema_sql = _SCHEMA_PATH.read_text()
    conn.executescript(schema_sql)
    conn.commit()
    return conn

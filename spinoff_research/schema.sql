-- Spin-off research prototype schema (SQLite).
-- Not part of Meridian's ChromaDB/JSONL persistence — this data must survive
-- restarts, unlike /tmp/meridian_chromadb, so it gets its own durable store.
-- See spinoff_research/db.py for the connection helper and CLAUDE.md's
-- "ChromaDB path is hardcoded" rule, which this intentionally does not follow
-- (this is not a ChromaDB collection).

PRAGMA foreign_keys = ON;

-- One row per real-world company (parent or spinco; a company can be both
-- across different transactions, e.g. a serial spinner).
CREATE TABLE IF NOT EXISTS companies (
    company_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    ticker          TEXT,                -- as-of-transaction ticker; may change later
    cik             TEXT,                -- SEC CIK, zero-padded to 10 digits; nullable until resolved
    sector          TEXT,
    industry        TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (name, ticker)
);

CREATE TABLE IF NOT EXISTS spinoff_transactions (
    transaction_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    source_row_number   INTEGER,          -- "Number" column from Rich's sheet, for traceability
    parent_company_id   INTEGER NOT NULL REFERENCES companies(company_id),
    spinoff_company_id  INTEGER NOT NULL REFERENCES companies(company_id),
    announcement_date   TEXT,             -- ISO date; needed to anchor annual-duration XBRL fields (see xbrl_service.py)
    spinoff_date        TEXT,             -- ISO date; nullable until known (regular-way trading date)
    status               TEXT NOT NULL DEFAULT 'pilot',  -- 'pilot' | 'active' | 'archived'
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (parent_company_id, spinoff_company_id)
);

-- Every SEC filing (or other source document) retrieved for a transaction.
CREATE TABLE IF NOT EXISTS documents (
    document_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id       INTEGER NOT NULL REFERENCES spinoff_transactions(transaction_id),
    company_id            INTEGER REFERENCES companies(company_id),  -- filer: parent or spinco
    cik                   TEXT,
    accession_number      TEXT,           -- SEC accession number, dashless; NULL for non-EDGAR sources
    filing_type            TEXT NOT NULL, -- '10-12B' | '10-K' | '10-Q' | '8-K' | 'DEF 14A' | 'FORM_4' | 'PRESS_RELEASE' | ...
    filing_date             TEXT,
    period_of_report         TEXT,
    sec_url                   TEXT,
    primary_document_name     TEXT,
    exhibit_number             TEXT,
    local_path                  TEXT,     -- cached raw file on disk
    content_hash                  TEXT,   -- sha256 of raw content, for de-dup
    ingestion_status                TEXT NOT NULL DEFAULT 'pending',  -- pending | downloaded | failed
    text_extraction_status           TEXT NOT NULL DEFAULT 'pending', -- pending | extracted | failed
    source_type                       TEXT NOT NULL,  -- matches field_data_dictionary.SourceType
    retrieved_at                       TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (accession_number, primary_document_name, exhibit_number)
);

-- Chunked/sectioned text extracted from a document, for citation and (later) retrieval.
CREATE TABLE IF NOT EXISTS document_sections (
    section_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     INTEGER NOT NULL REFERENCES documents(document_id),
    section_title   TEXT,
    chunk_index     INTEGER NOT NULL,
    char_start      INTEGER,
    char_end        INTEGER,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per (transaction, field) extraction attempt. Multiple rows per
-- field are expected (re-extraction, different methods tried).
CREATE TABLE IF NOT EXISTS field_extraction_runs (
    run_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id    INTEGER NOT NULL REFERENCES spinoff_transactions(transaction_id),
    field_key          TEXT NOT NULL,      -- matches field_data_dictionary.FieldDefinition.field_key
    extraction_method    TEXT NOT NULL,    -- 'xbrl' | 'filing_metadata' | 'ai_assisted' | 'calculated' | 'manual'
    run_at                TEXT NOT NULL DEFAULT (datetime('now')),
    model_used              TEXT,          -- e.g. 'gpt-4o', NULL for non-AI methods
    status                    TEXT NOT NULL -- see field_data_dictionary status model
);

-- The current/approved value for a (transaction, field) — one logical row
-- per field per transaction, but preserves the original extracted value
-- even after human correction (never overwritten, only superseded).
CREATE TABLE IF NOT EXISTS field_values (
    field_value_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id       INTEGER NOT NULL REFERENCES spinoff_transactions(transaction_id),
    field_key             TEXT NOT NULL,
    run_id                  INTEGER REFERENCES field_extraction_runs(run_id),
    raw_value                 TEXT,        -- as found in source text, unmodified
    normalized_value            TEXT,      -- coerced to field's data_type
    numeric_value                  REAL,   -- populated when data_type is numeric/currency/ratio/percent
    unit                             TEXT,
    currency                          TEXT,
    scale                               TEXT,  -- 'ones' | 'thousands' | 'millions' | 'billions'
    reporting_period                     TEXT,
    as_of_date                             TEXT,
    value_basis                             TEXT,  -- 'historical' | 'pro_forma' | 'guidance' | 'calculated'
    confidence                                REAL,
    status                                      TEXT NOT NULL,  -- see status model below
    is_original_extraction                        INTEGER NOT NULL DEFAULT 1,  -- 0 once superseded by a review correction
    superseded_by_field_value_id                    INTEGER REFERENCES field_values(field_value_id),
    created_at                                        TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Every source citation backing a field_value. A single field_value may
-- have multiple sources (brief: "Some judgment fields may require multiple
-- citations").
CREATE TABLE IF NOT EXISTS field_sources (
    field_source_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    field_value_id     INTEGER NOT NULL REFERENCES field_values(field_value_id),
    document_id          INTEGER REFERENCES documents(document_id),
    section_id             INTEGER REFERENCES document_sections(section_id),
    supporting_excerpt        TEXT,
    page_number                 INTEGER,
    reasoning_summary             TEXT,   -- concise evidence explanation; never hidden chain-of-thought
    created_at                     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Human review actions on a field_value. Preserves audit history —
-- never deletes or overwrites a prior review row.
CREATE TABLE IF NOT EXISTS field_reviews (
    review_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    field_value_id      INTEGER NOT NULL REFERENCES field_values(field_value_id),
    action                 TEXT NOT NULL,  -- 'approve' | 'edit' | 'reject' | 'mark_not_found'
    corrected_value           TEXT,
    reviewer_notes               TEXT,
    reviewed_by                     TEXT,
    reviewed_at                       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Fields that cannot be determined at extraction time (dividend-within-12mo,
-- insider clustering, TSR, etc.) — models the "not yet determinable" state
-- and what would need to happen to resolve it later.
CREATE TABLE IF NOT EXISTS scheduled_field_requirements (
    requirement_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id         INTEGER NOT NULL REFERENCES spinoff_transactions(transaction_id),
    field_key                TEXT NOT NULL,
    observation_window_days    INTEGER NOT NULL,
    earliest_determinable_date   TEXT NOT NULL,  -- computed: spinoff_date + window
    required_data_source           TEXT NOT NULL,
    requires_manual_review           INTEGER NOT NULL DEFAULT 0,
    status                             TEXT NOT NULL DEFAULT 'not_yet_determinable',
    checked_at                           TEXT,
    UNIQUE (transaction_id, field_key)
);

CREATE INDEX IF NOT EXISTS idx_field_values_txn_field ON field_values(transaction_id, field_key);
CREATE INDEX IF NOT EXISTS idx_documents_txn ON documents(transaction_id);
CREATE INDEX IF NOT EXISTS idx_field_sources_value ON field_sources(field_value_id);
CREATE INDEX IF NOT EXISTS idx_scheduled_txn_field ON scheduled_field_requirements(transaction_id, field_key);

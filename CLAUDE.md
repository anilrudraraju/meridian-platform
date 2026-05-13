# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
- **App:** Meridian Intelligence Platform — 10-layer AI wealth management platform
- **Case Study:** Global Fiscal Group ($12.8B AUM)
- **Stack:** Python + Streamlit, single `app.py` file (~1,550 lines)
- **Live URL:** https://meridian-platform.streamlit.app
- **Repo:** https://github.com/anilrudraraju/meridian-platform
- **Deploy:** Every push to `main` triggers auto-redeploy on Streamlit Cloud (~60s)

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501 and enter your OpenAI key in the sidebar
```

No build step, lint config, or test suite exists. The running app is the artifact.

---

## Current Layer Status

| Layer | Status | `active_layer` value | Label |
|-------|--------|---------------------|-------|
| 1 | ✅ BUILT | `"guardrails"` | 🛡️ Layer 1 — Guardrails & Prompts |
| 2 | ✅ BUILT | `"portfolio"` | 📈 Layer 2 — Portfolio Dashboard |
| 3 | ✅ BUILT | `"rag"` | 📄 Layer 3 — Document RAG |
| 4 | ✅ BUILT | `"finetune"` | 🔬 Layer 4 — Fine-Tuning & Evaluation |
| 5–10 | 🔜 | follow same pattern | — |

---

## Navigation Architecture

The app uses **sidebar buttons + `st.session_state.active_layer`**, not `st.tabs()`. Each layer renders inside an `if st.session_state.active_layer == "<value>":` block at the bottom of `app.py`. Sidebar buttons set `active_layer` then call `st.rerun()`.

**Do not use `st.tabs()` or numbered tab variables** — this previously crashed the deployment.

Adding a new layer means:
1. Add a sidebar button that sets `st.session_state.active_layer = "<new_value>"`
2. Add a corresponding `if st.session_state.active_layer == "<new_value>":` UI block

---

## CRITICAL CODING RULES — Always Follow

1. **Navigation uses named `active_layer` strings.**
   - ✅ `st.session_state.active_layer == "guardrails"` / `"portfolio"` / `"rag"` / `"finetune"`
   - ❌ Never `tab1`, `tab2`, or `st.tabs()`

2. **Classes must match notebook source exactly.**
   - `FinancialPromptEngine`, `FinancialGuardrails`, `DocumentProcessor`, `RAGSystem`
   - Do NOT rename methods or change signatures

3. **Dataclass fields are frozen** — never add/remove fields from:
   - `PromptResult`, `GuardrailResult`, `SearchResult`, `RAGResponse`

4. **Always wrap LLM calls** through `FinancialGuardrails.safe_execute()` in the UI

5. **ChromaDB path is hardcoded:** `/tmp/meridian_chromadb` — never use `tempfile` or `os.path.expanduser`

6. **ChromaDB IDs use MD5 hash:**
   ```python
   hashlib.md5(f"{source}__chunk_{id}".encode()).hexdigest()
   ```

7. **Embeddings in batches of 100** — current limit in `RAGSystem._embed()` (raised from 20 for throughput)

8. **ChromaDB upsert in batches of 100** — never upsert all at once

9. **RAG temperature = 0** — deterministic financial answers only

10. **No `tempfile` import** — use `/tmp/meridian_chromadb` as a hardcoded string

---

## Tech Stack (DO NOT CHANGE)

```
Language:     Python 3.9+
Frontend:     Streamlit >= 1.32.0
LLM:          OpenAI GPT-4 / GPT-4o (via openai >= 1.12.0)
Embeddings:   text-embedding-ada-002 (1536 dims) — never change this
Vector DB:    ChromaDB >= 1.0.0 (PersistentClient)
Market Data:  yfinance >= 0.2.36
PDF Parsing:  pypdf >= 4.0.0 (DEFAULT fast path) + pdfplumber >= 0.10.0 (opt-in, table-aware)
SEC Filings:  SEC EDGAR API (free, no key)
HTTP:         requests >= 2.31.0
Data:         pandas >= 2.0.0, numpy >= 1.24.0
Hosting:      Streamlit Community Cloud
```

## OpenAI Models
```python
"gpt-4o"                 # portfolio analysis (default)
"gpt-4"                  # RAG Q&A (temperature=0 — deterministic)
"gpt-4o-mini"            # guardrails prompt demos (cost saving)
"text-embedding-ada-002" # ALL embeddings — never change
```

---

## Class Reference

### Layer 1 — `FinancialPromptEngine` & `FinancialGuardrails`
```python
@dataclass
class PromptResult:
    prompt: str; response: str; model: str
    tokens_used: int; cost_estimate: float; timestamp: str
    technique: str  # "zero-shot","few-shot","chain-of-thought","role-based","react"

@dataclass
class GuardrailResult:
    passed: bool; message: str; violations: List[str]
    modified_content: Optional[str] = None

class FinancialPromptEngine:
    def portfolio_risk_analysis(self, portfolio_data: str) -> PromptResult      # zero-shot
    def portfolio_report_fewshot(self, portfolio_data: str) -> PromptResult     # few-shot
    def tax_loss_harvesting_cot(self, holdings_data: str) -> PromptResult       # chain-of-thought
    def client_communication(self, situation: str, client_type: str) -> PromptResult  # role-based
    def market_commentary_react(self, market_event: str) -> PromptResult        # react

class FinancialGuardrails:
    def validate_input(self, user_input: str) -> GuardrailResult
    def validate_output(self, ai_output: str) -> GuardrailResult
    def safe_execute(self, prompt_engine, prompt_function, *args, **kwargs) -> Tuple[bool, PromptResult]
```

### Layer 3 — `DocumentProcessor` & `RAGSystem`
```python
@dataclass
class SearchResult:
    content: str; source: str
    relevance_score: float   # cosine similarity 0–1
    metadata: Dict

@dataclass
class RAGResponse:
    question: str; answer: str
    sources: List[SearchResult]
    confidence: str  # "High" (>0.75), "Medium" (>0.60), "Low"

class DocumentProcessor:
    # chunk_size=2000, chunk_overlap=400
    # chunk_text pre-scans for SEC "ITEM N" headers and tags each chunk with a "section" metadata field
    def chunk_text(self, text: str, source: str) -> List[Dict]
    def load_from_text(self, text: str, source: str) -> List[Dict]
    def load_from_pdf_bytes(self, pdf_bytes: bytes, source: str, table_aware: bool = False) -> List[Dict]
    #   table_aware=False → pypdf (fast, ~15-30s per file, default)
    #   table_aware=True  → pdfplumber (2+ min for large 10-Qs, preserves table structure)
    def load_from_txt_bytes(self, txt_bytes: bytes, source: str) -> List[Dict]

class RAGSystem:
    def index_documents(self, chunks: List[Dict]) -> None
    def search(self, query: str, k: int = 5) -> List[SearchResult]
    def answer_question(self, question: str, k: int = 5) -> RAGResponse  # temperature=0; UI calls with k=10; filters results < 0.55 similarity
    def analyze_risk_factors(self, company: str) -> RAGResponse
    def summarize_earnings(self, company: str, quarter: str) -> RAGResponse
    def clear(self) -> None   # delete + recreate ChromaDB collection
    def count(self) -> int    # number of chunks currently indexed
```

### Layer 4 — `FinancialEvaluator`
```python
# Model constants (top of app.py)
BASE_MODEL       = "gpt-3.5-turbo-0125"
FINE_TUNED_MODEL = "ft:gpt-3.5-turbo-0125:personal::DZTJSppd"

# Sentence-transformer model is loaded via @st.cache_resource to avoid reload on every rerun
@st.cache_resource
def load_evaluator():  # returns (SentenceTransformer, RougeScorer)

class FinancialEvaluator:
    def evaluate_semantic_similarity(self, pred: str, ref: str) -> float  # cosine sim via all-MiniLM-L6-v2
    def check_compliance(self, text: str) -> float  # checks "past performance" + "does not guarantee" → 0.0/0.5/1.0
```
**Training data:** `training_data.jsonl` at repo root — 56 examples used to fine-tune `FINE_TUNED_MODEL`.

---

### Deployment Helpers (not in notebooks)
```python
class MarketDataFetcher:
    def fetch_portfolio(self, holdings: Dict[str, float]) -> Tuple[List[Dict], float, List[str]]

def fetch_edgar_filing(ticker: str, form_type: str = "10-K") -> Tuple[bool, str, str]:
    # char_cap = 300,000 chars
    # 10-Qs work well; 10-Ks often exceed cap — use PDF upload instead

def fetch_xbrl_facts(ticker: str) -> Tuple[bool, Dict, str]:
    # Returns {metric_label: [{value, period_end, period_start, form, filed, period}]}
    # Source: SEC XBRL Company Facts API — exact numbers, no embeddings needed
    # Metrics: Revenue, Net Income, Operating Income, Gross Profit, Total Assets,
    #          Total Liabilities, Stockholders Equity, Operating Cash Flow,
    #          Cash & Equivalents, EPS (Diluted), Long-Term Debt, R&D Expense

def _fmt_xbrl(value: float, label: str) -> str:
    # Scales raw USD values to T/B/M strings; EPS shown as $X.XX
```

**Dual-store pattern (Layer 3):**
- **Quantitative** questions (revenue, net income, EPS) → Step 4 XBRL fetch — exact numbers, no RAG
- **Qualitative** questions (why revenue declined, risk factors, MD&A) → Steps 1–3 RAG pipeline

---

## ChromaDB Config
```python
CHROMA_PERSIST_DIR = "/tmp/meridian_chromadb"  # hardcoded — do not change
CHROMA_COLLECTION  = "meridian_docs"
# Created with: metadata={"hnsw:space": "cosine"}
# Distance-to-similarity: score = 1 - (distance / 2)
```
**Note:** `/tmp` resets after ~1hr inactivity on Streamlit Cloud. Re-indexing required after wake.

---

## Layer 5 — Next Layer (Responsible AI & Safety)

**Navigation value:** `"responsible_ai"` — label "🧭 Layer 5 — Responsible AI & Safety"

Planned features: bias detection, hallucination guard, audit logging.

Adding it follows the same pattern as all prior layers:
1. Add a sidebar button setting `st.session_state.active_layer = "responsible_ai"`
2. Add a corresponding `if st.session_state.active_layer == "responsible_ai":` UI block

---

## API Key
Stored in Streamlit Cloud Secrets as `OPENAI_API_KEY`. Load with:
```python
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key
except:
    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
```

# Meridian Intelligence Platform — Claude Code Context

## Project Overview
- **App:** Meridian Intelligence Platform — 10-layer AI wealth management platform
- **Case Study:** Global Fiscal Group ($12.8B AUM)
- **Stack:** Python + Streamlit, single `app.py` file (~1,100 lines)
- **Live URL:** https://meridian-platform.streamlit.app
- **Repo:** https://github.com/anilrudraraju/meridian-platform
- **Local path:** `~/Desktop/meridian-platform`
- **Deploy:** Every push to `main` triggers auto-redeploy on Streamlit Cloud (~60s)

---

## Current Layer Status
| Layer | Status | Tab Variable | Label |
|-------|--------|-------------|-------|
| 1 | ✅ BUILT | `tab_guardrails` | 🛡️ Layer 1 — Guardrails & Prompts |
| 2 | ✅ BUILT | `tab_portfolio` | 📈 Layer 2 — Portfolio Dashboard |
| 3 | ✅ BUILT | `tab_rag` | 📄 Layer 3 — Document RAG |
| 4 | 🔜 NEXT | `tab_finetune` | 🔬 Layer 4 — Fine-Tuning & Evaluation |
| 5–10 | 🔜 | follow same pattern | — |

---

## CRITICAL CODING RULES — Always Follow

1. **Tab variables are NAMED, not numbered.**
   - ✅ `tab_guardrails`, `tab_portfolio`, `tab_rag`, `tab_finetune`
   - ❌ Never `tab1`, `tab2`, `tab3` — this crashed deployment before

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

7. **Embeddings in batches of 20** — OpenAI ada-002 API limit

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
PDF Parsing:  pdfplumber >= 0.10.0 (PRIMARY) + pypdf >= 4.0.0 (fallback)
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

### Week 1 — `FinancialPromptEngine` & `FinancialGuardrails`
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

### Week 3 — `DocumentProcessor` & `RAGSystem`
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
    confidence: str  # "High" (>0.80), "Medium" (>0.70), "Low"

class DocumentProcessor:
    # chunk_size=1000, chunk_overlap=200 — DO NOT CHANGE (assignment spec)
    def chunk_text(self, text: str, source: str) -> List[Dict]
    def load_from_text(self, text: str, source: str) -> List[Dict]
    def load_from_pdf_bytes(self, pdf_bytes: bytes, source: str) -> List[Dict]  # pdfplumber primary
    def load_from_txt_bytes(self, txt_bytes: bytes, source: str) -> List[Dict]

class RAGSystem:
    def index_documents(self, chunks: List[Dict]) -> None
    def search(self, query: str, k: int = 5) -> List[SearchResult]
    def answer_question(self, question: str, k: int = 5) -> RAGResponse  # temperature=0
    def analyze_risk_factors(self, company: str) -> RAGResponse
    def summarize_earnings(self, company: str, quarter: str) -> RAGResponse
```

### Deployment Helpers (not in notebooks)
```python
class MarketDataFetcher:
    def fetch_portfolio(self, holdings: Dict[str, float]) -> Tuple[List[Dict], float, List[str]]

def fetch_edgar_filing(ticker: str, form_type: str = "10-K") -> Tuple[bool, str, str]:
    # char_cap = 300,000 chars
    # 10-Qs work well; 10-Ks often exceed cap — use PDF upload instead
```

---

## ChromaDB Config
```python
CHROMA_PERSIST_DIR = "/tmp/meridian_chromadb"  # hardcoded — do not change
CHROMA_COLLECTION  = "meridian_docs"
# Created with: metadata={"hnsw:space": "cosine"}
# Similarity: 1 - (distance / 2)
```
**Note:** `/tmp` resets after ~1hr inactivity on Streamlit Cloud. Re-indexing required after wake.

---

## Known Issues & Fixes Applied

| Bug | Fix |
|-----|-----|
| `chunk_text` infinite loop | `step = max(1, chunk_size - chunk_overlap)` |
| PDF silent failure | Per-page try/except in `load_from_pdf_bytes` |
| ChromaDB upsert size crash | Batch upserts of 100 |
| ChromaDB invalid IDs | MD5 hash for all IDs |
| OpenAI token limit on embeddings | Truncate chunks to 6,000 chars before embedding |
| EDGAR truncation too aggressive | Raised cap to 300,000 chars |
| Financial tables misread | Switched to pdfplumber (table-aware) |
| Non-deterministic RAG answers | Set `temperature=0` in `answer_question()` |
| Tab variable crash on deploy | Named tab vars (`tab_guardrails` not `tab1`) |
| protobuf conflict | `protobuf>=3.20.0,<4.0.0` in requirements.txt |
| ChromaDB path permission error | Changed to `/tmp/meridian_chromadb` |
| API key not loading | `try: st.secrets["OPENAI_API_KEY"]` pattern |

---

## Week 4 — Next Layer (Fine-Tuning & Evaluation)

**New tab:** `tab_finetune` — "🔬 Layer 4 — Fine-Tuning & Evaluation"

**New classes:**
```python
class FinancialEvaluator:
    def evaluate_semantic_similarity(self, pred: str, ref: str) -> float
    def check_compliance(self, text: str) -> float  # 0.0 to 1.0
```

**New packages to add to requirements.txt:**
```
rouge-score
sentence-transformers
scikit-learn
```

---

## File Structure
```
meridian-platform/
├── app.py              # Single Streamlit app — all layers here
├── requirements.txt    # All dependencies pinned
├── CLAUDE.md           # This file — Claude Code context
└── README.md           # Setup and deployment instructions
```

---

## API Key
Stored in Streamlit Cloud Secrets as `OPENAI_API_KEY`. Load with:
```python
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except:
    openai.api_key = os.getenv("OPENAI_API_KEY")
```

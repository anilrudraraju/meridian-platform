# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
- **App:** Meridian Intelligence Platform — 10-layer AI wealth management platform
- **Case Study:** Global Fiscal Group ($12.8B AUM)
- **Stack:** Python + Streamlit, single `app.py` (~2,300 lines) + `core/` modules
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

No build step or lint config. Tests cover pure helper functions only (no Streamlit UI, no API calls):

```bash
python3 test_app.py          # runs ~40 unit tests; all should pass with no keys
```

**Deployment rule:** Always `git push` immediately after committing and verify with `git log --oneline origin/main..HEAD` (should be empty) before claiming anything is deployed. Streamlit Cloud deploys from GitHub — local commits alone change nothing.

**APP_VERSION** is defined at the top of `app.py` (`YYYY-MM-DD-vN` format). Bump it on every meaningful commit so the sidebar reflects what's deployed.

---

## Layer Status

| Layer | `active_layer` value | Core module | Label |
|-------|---------------------|-------------|-------|
| 1 ✅ | `"guardrails"` | `core/prompts.py`, `core/guardrails.py` | 🛡️ Layer 1 — Guardrails & Prompts |
| 2 ✅ | `"portfolio"` | `core/market.py` | 📈 Layer 2 — Portfolio Dashboard |
| 3 ✅ | `"rag"` | `core/rag.py`, `core/edgar.py`, `core/chunking.py` | 📄 Layer 3 — Document RAG |
| 4 ✅ | `"finetune"` | `core/evaluation.py` | 🔬 Layer 4 — Fine-Tuning & Evaluation |
| 5 ✅ | `"responsible_ai"` | `core/safety.py` | 🧭 Layer 5 — Responsible AI & Safety |
| 6 ✅ | `"agents"` | `core/react_agent.py` | 🤖 Layer 6 — Autonomous ReAct Agents |
| 7 ✅ | `"multi_agent"` | `core/crew_agents.py` | 🤝 Layer 7 — Multi-Agent Collaboration |
| 8 ✅ | `"rebalancing"` | `core/rebalancing_workflow.py` | ⚖️ Layer 8 — Stateful Rebalancing |
| 9 ✅ | `"committee"` | `core/investment_committee.py` | 🏛️ Layer 9 — Investment Committee |
| 10 ✅ | `"integrated"` | `core/cost.py` | 🏗️ Layer 10 — Integrated Platform |
| Special Situations Lab ✅ | `"spinoff_lab"` | `special_situations/spinoff_lab.py` | 🔬 Special Situations Lab |

---

## Navigation Architecture

The app uses **sidebar buttons + `st.session_state.active_layer`**, not `st.tabs()`. Each layer renders inside an `if st.session_state.active_layer == "<value>":` block at the bottom of `app.py`.

**Never use `st.tabs()` or numbered tab variables** — this previously crashed the deployment.

Adding a new layer:
1. Add a sidebar button entry to the loop in `app.py` (around line 80)
2. Add a corresponding `if st.session_state.active_layer == "<value>":` block before the Special Situations Lab section

**Special Situations Lab** sits outside the numbered layer loop. Its sidebar button is added separately (after the divider following Layer 10, around line 101). It is rendered via `spinoff_lab.render()` from `special_situations/spinoff_lab.py`, which itself imports from the `spinoff/` subpackage (`models`, `greenblatt_scorecard`, `promise_tracker`, `thesis_tracker`, `cost_tracker`). Sample CSV data lives in `data/spinoffs/`.

---

## Critical Coding Rules

1. **Navigation uses named `active_layer` strings** — never `tab1`, `tab2`, or `st.tabs()`

2. **Dataclass fields are frozen** — never add/remove fields from `PromptResult`, `GuardrailResult`, `SearchResult`, `RAGResponse` (defined in `core/dataclasses.py`)

3. **ChromaDB path is hardcoded:** `/tmp/meridian_chromadb` — never `tempfile` or `os.path.expanduser`

4. **ChromaDB IDs use MD5 hash:** `hashlib.md5(f"{source}__chunk_{id}".encode()).hexdigest()`

5. **Embeddings in batches of 100** — `RAGSystem._embed()` limit

6. **ChromaDB upsert in batches of 100** — never upsert all at once

7. **RAG temperature = 0** — deterministic financial answers only

8. **Token optimisation on every LLM call:**
   - Keep system prompts under ~50 words per agent
   - Always set `max_tokens` (400–600 for agents, 200–300 for structured output)
   - Truncate inter-agent context to 250–300 chars per message
   - Include word limits ("under 200 words") in user prompts
   - Use plain Python for threshold checks and math — never an LLM call

---

## Tech Stack (Do Not Change)

```
Language:     Python 3.9+
Frontend:     Streamlit >= 1.32.0
LLM:          OpenAI GPT-4o / GPT-4 / GPT-3.5 (via openai >= 1.12.0)
Embeddings:   text-embedding-ada-002 (1536 dims) — never change
Vector DB:    ChromaDB >= 1.0.0 (PersistentClient, cosine space)
Market Data:  yfinance >= 0.2.36
Multi-agent:  crewai >= 0.80.0 (Layer 7 only)
State machine:langgraph >= 0.2.0 (Layer 8 only)
PDF Parsing:  pypdf >= 4.0.0 (default) + pdfplumber >= 0.10.0 (table-aware)
SEC Filings:  SEC EDGAR API (free, no key)
Hosting:      Streamlit Community Cloud
```

### OpenAI Model Usage
```python
"gpt-4o"                 # Layers 6, 7, 9 — agent reasoning (default)
"gpt-4"                  # Layer 3 RAG Q&A (temperature=0)
"gpt-4o-mini"            # Layer 1 prompt demos (cost saving)
"gpt-3.5-turbo-0125"     # Layer 4 base model comparison
"ft:gpt-3.5-turbo-0125:personal::DZTJSppd"  # Layer 4 fine-tuned model
"text-embedding-ada-002" # ALL embeddings — never change
```

---

## Core Module Reference

### Layer 1 — `core/prompts.py`, `core/guardrails.py`
```python
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

### Layer 3 — `core/rag.py`, `core/edgar.py`, `core/chunking.py`
```python
class DocumentProcessor:
    def process_filing(self, source, company, ticker, cik, form_type, fiscal_year,
                       quarter=None, period_end=None, pdf_bytes=None, text=None) -> List[Dict]
    # Use this for all SEC filings. Legacy: load_from_text(), load_from_pdf_bytes()

class RAGSystem:
    def index_documents(self, chunks: List[Dict]) -> None
    def search(self, query: str, k: int = 5, where: dict = None) -> List[SearchResult]
    def answer_question(self, question: str, k: int = 5) -> RAGResponse  # UI calls k=10
    def clear(self) -> None
    def count(self) -> int

# EDGAR helpers in core/edgar.py
fetch_edgar_filing(ticker, form_type="10-K", target_year=None)  # → (ok, text, desc, cik, company)
fetch_xbrl_facts(ticker)  # → (ok, dict, msg) — exact financials, no embeddings needed
build_chroma_filter(filters)  # builds ChromaDB $and filter
route_query(question)  # → "structured" | "narrative" | "both"
```

**Single-company constraint:** only one company's data can be in ChromaDB at a time. Switching tickers calls `_clear_for_new_company()`, which wipes `rag_system`, `all_chunks`, `loaded_docs`, and `xbrl_by_ticker` from session state. Never mix data from multiple companies.

**Dual-store pattern:** quantitative questions (revenue, EPS) → XBRL; qualitative (risk factors, MD&A) → RAG.

```python
# Unified retrieval (core/rag.py, imported in app.py)
retrieve(question, ticker, fiscal_year, form_type, quarter, rag, xbrl_facts)
# Routes to XBRL for structured queries, RAG for narrative — returns combined context string
```

**Section patterns** used by `_split_into_sections()` live in `core/constants.py` and are imported at the top of `app.py`: `SECTION_PATTERNS_10K`, `SECTION_PATTERNS_10Q`, `SECTION_PATTERNS_FORM10_EXTRA`, `STATEMENT_PATTERNS`, `MDNA_SUBSECTION_PATTERNS`, `STRUCTURED_SIGNALS`, `NARRATIVE_SIGNALS`.

**ChromaDB config:**
```python
CHROMA_PERSIST_DIR = "/tmp/meridian_chromadb"   # resets after ~1hr on Streamlit Cloud
CHROMA_COLLECTION  = "meridian_docs"            # cosine similarity, score = 1 - distance/2
```

**Section-aware chunking** (`core/chunking.py`): `_split_into_sections()` splits filings into 9 named sections (business, risk_factors, mdna, financial_stmts, footnotes, quantitative, controls, legal, default), each with its own chunking strategy and size/overlap.

### Layer 4 — `core/evaluation.py`
```python
class FinancialEvaluator:
    def evaluate_semantic_similarity(self, pred: str, ref: str) -> float  # sentence-transformers cosine sim
    def check_compliance(self, text: str) -> float                       # fraction of required disclaimer phrases present
```
Compares `BASE_MODEL` ("gpt-3.5-turbo-0125") vs `FINE_TUNED_MODEL` (the `ft:...` id above). `load_evaluator()` is `@st.cache_resource`-cached since it loads a SentenceTransformer.

### Layer 5 — `core/safety.py`
```python
class PIIScanner      # PII + prompt injection + blocked-topic detection
class BiasDetector    # demographic bias testing across prompt variants
class AuditLogger     # append-only JSONL at /tmp/meridian_audit.jsonl
```

### Layer 6 — `core/react_agent.py`
Direct OpenAI tool-calling (NOT LangChain). Three tools registered in `_TOOL_FUNCTIONS` / `_TOOL_SCHEMAS`:
`GetStockPrice` · `GetPortfolioValue` · `CheckPortfolioAlerts`

```python
class PortfolioReActAgent:
    def run(self, task: str, on_step=None) -> dict
    # on_step(thought, tool_name, observation) — called after each tool execution
    # Returns {output, steps, iterations, prompt_tokens, completion_tokens, cost_usd}

class SafeAgentExecutor:   # wraps agent with FinancialGuardrails input/output validation
class AgentEvaluator:      # runs 4 predefined test cases, returns accuracy + avg_steps
```

The ReAct loop: `messages` list grows each iteration — LLM appends its reasoning + tool call, your code executes the tool and appends the result, repeat until LLM returns no `tool_calls`.

### Layer 7 — `core/crew_agents.py`
CrewAI sequential crew: Research Analyst → Risk Specialist → Portfolio Manager.

```python
class PortfolioAnalysisCrew:
    def run(self, holdings: dict, on_task=None) -> dict
    # holdings: {ticker: decimal_weight} e.g. {"AAPL": 0.40, "MSFT": 0.35}
    # on_task(agent_role, output_text) called after each task
    # Returns {output, agent_outputs, holdings}
```

Tools: `GetPortfolioData` (batch price/PE/sector fetch) · `CalculatePortfolioRisk` (weighted volatility via `TICKER:weight` format).

### Layer 8 — `core/rebalancing_workflow.py`
LangGraph StateGraph with `interrupt_before=["human_approval"]`.

```python
class RebalancingWorkflow:
    def analyze(self, portfolio, target_allocation, approval_threshold=1_000_000,
                thread_id="default", on_node=None) -> dict
    # on_node(node_name, updates, accumulated_state) — streams live node updates
    # Returns dict with status: "completed" | "pending_approval" | "no_action_needed" | "rejected"

    def complete(self, approved: bool, thread_id="default") -> dict
    # Call after analyze() returns status="pending_approval"
```

Graph: `check_drift` → [drift > 5%] → `generate_trades` → `optimize_tax` → `check_approval` → [needs approval] → ⏸ `human_approval` → `execute_trades` / `rejection`.

All nodes are pure Python — no LLM calls in this layer.

### Layer 9 — `core/investment_committee.py`
Three-agent debate via MessageBus. Direct OpenAI calls (not CrewAI).

```python
class MessageBus:
    def send(self, message: Message) -> None
    def get_by_round(self, round_num: int) -> List[Message]
    def context_for(self, agent_role: str, before_round: int, max_chars=300) -> str
    def full_transcript(self, max_chars=250) -> str

class InvestmentCommittee:
    def run(self, proposal: str, on_message=None) -> dict
    # on_message(round_num, agent_role, icon, message_type, content)
    # Returns {proposal, bus, votes, tally, outcome, agent_outputs, message_count}
    # outcome: "APPROVED" | "REJECTED" | "MODIFICATION REQUIRED" | "NO CONSENSUS — REQUIRES FURTHER DISCUSSION"
```

Agents defined in `_AGENTS` list (Growth Specialist · Value Specialist · Chief Risk Officer). Add new agents by appending to this list.

### Cost Tracking — `core/cost.py`
```python
log_call(model, prompt_tokens, completion_tokens, cost_usd, technique, caller)
daily_spend() -> float
check_budget() -> (under_budget, spent, cap)   # cap set via MERIDIAN_DAILY_CAP env var (default $5)
read_log() -> list[dict]   # all entries, newest first
```
Log written to `/tmp/meridian_cost_log.jsonl`. Call `log_call()` at the end of every LLM-using function.

---

## API Key Pattern
```python
try:
    api_key = st.secrets["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key
except:
    api_key = st.text_input("🔑 OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
```

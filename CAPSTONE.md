# Meridian Intelligence Platform — Capstone Presentation

**Case Study:** Global Fiscal Group | **AUM:** $12.8B | **Stack:** Python · Streamlit · OpenAI

---

## 1. Business Problem

Wealth management firms face three compounding pressures:
- **Scale**: Advisors cannot personally monitor hundreds of portfolios in real time
- **Speed**: Market events require analysis and action in minutes, not days
- **Compliance**: Every recommendation must be auditable and within regulatory guardrails

Traditional software can automate rules, but cannot reason. The Meridian platform introduces AI reasoning at every layer — from document Q&A to autonomous agents to committee-style deliberation — while keeping humans in control of high-stakes decisions.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  MERIDIAN INTELLIGENCE PLATFORM              │
├──────────────┬──────────────────────────────────────────────┤
│   ADVISOR UI │  Streamlit sidebar navigation (single app.py) │
├──────────────┴──────────────────────────────────────────────┤
│  L9 Investment Committee  │  L8 Stateful Rebalancing         │
│  MessageBus · 3-round     │  LangGraph · drift detection     │
│  debate · consensus vote  │  · tax optimisation · approval   │
├───────────────────────────┼─────────────────────────────────┤
│  L7 Multi-Agent Crew      │  L6 Autonomous ReAct Agent       │
│  CrewAI · Research +      │  OpenAI tool-calling · ReAct     │
│  Risk + Portfolio Mgr     │  loop · portfolio monitor        │
├───────────────────────────┼─────────────────────────────────┤
│  L5 Responsible AI        │  L4 Fine-Tuning & Evaluation     │
│  PII scanner · bias       │  GPT-3.5 fine-tuned · ROUGE      │
│  detection · audit log    │  · semantic similarity scoring   │
├───────────────────────────┼─────────────────────────────────┤
│  L3 Document RAG          │  L2 Portfolio Dashboard          │
│  ChromaDB · EDGAR 10-K    │  yfinance · live valuation       │
│  · section-aware chunks   │  · MarketDataFetcher             │
├───────────────────────────┴─────────────────────────────────┤
│  L1 Prompt Engineering Foundation                            │
│  5 techniques: zero-shot · few-shot · CoT · role · ReAct     │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
    OpenAI API           ChromaDB             SEC EDGAR API
  (GPT-4o, GPT-4,       /tmp persist         (free, no key)
   GPT-3.5, ada-002)
```

---

## 3. Layer-by-Layer Summary

| Layer | What it does | Key technique | Framework |
|-------|-------------|---------------|-----------|
| 1 | Prompt engineering demos | Zero-shot, few-shot, CoT, role-based, ReAct | OpenAI |
| 2 | Live portfolio valuation | yfinance batch fetch | yfinance / pandas |
| 3 | 10-K / 10-Q document Q&A | Section-aware RAG, XBRL dual-store | ChromaDB |
| 4 | Base vs fine-tuned model comparison | ROUGE + semantic similarity | OpenAI fine-tune |
| 5 | PII detection, bias testing, audit log | Regex + LLM-based scanning | Custom |
| 6 | Autonomous portfolio monitoring | ReAct tool-calling loop | OpenAI |
| 7 | 3-agent portfolio analysis crew | Sequential multi-agent | CrewAI |
| 8 | Portfolio rebalancing with human gate | State machine + interrupt | LangGraph |
| 9 | Investment committee debate | MessageBus 3-round protocol | OpenAI |
| 10 | Integrated system | All layers unified in one app | Streamlit |

---

## 4. End-to-End Client Workflow

A complete client portfolio review chains five layers automatically:

```
Client portfolio input
        │
        ▼
L6: ReAct Agent monitors for alerts (price movements > 5%)
        │
        ▼
L7: CrewAI crew runs Research → Risk → PM analysis
        │
        ▼
L8: LangGraph checks drift → generates trades → tax optimisation
        │  (if trade value > $1M)
        ▼
L9: Investment Committee debates and votes on the rebalancing proposal
        │
        ▼
L5: Guardrails validate output → PII scan → audit log entry
        │
        ▼
L1: Role-based prompt drafts client communication
```

Human advisor sees the full audit trail and approves or rejects at the L8 gate.

---

## 5. Technical Highlights

**Section-aware RAG (Layer 3)**
SEC 10-K filings are split into 9 named sections (business, risk factors, MD&A, financials, footnotes, etc.) with different chunking strategies per section. A query router decides whether to use XBRL (exact numbers) or RAG (narrative). This produces significantly more accurate answers than naive sliding-window chunking.

**Human-in-the-loop (Layer 8)**
LangGraph's `interrupt_before` pauses the state machine before the approval node. The UI resumes the same graph thread after the advisor clicks Approve or Reject — state is fully persisted across the pause via MemorySaver checkpointing.

**MessageBus debate protocol (Layer 9)**
Three agents with distinct investment philosophies (Growth, Value, Risk) read each other's positions between rounds. Round 2 context is truncated to 300 chars/message to limit token cost while preserving key arguments. Majority vote determines outcome.

**Token optimisation (all layers)**
- System prompts trimmed to < 50 words per agent
- `max_tokens` hard-capped on every LLM call
- Inter-agent context truncated, not passed in full
- gpt-4o-mini used for guardrail demos (10× cheaper than gpt-4o)
- Pure-logic nodes (drift calculation, threshold checks) run as plain Python — no LLM

---

## 6. Technology Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.9+ |
| UI | Streamlit ≥ 1.32 |
| LLM | OpenAI GPT-4o, GPT-4, GPT-3.5-turbo |
| Embeddings | text-embedding-ada-002 (1536 dims) |
| Vector DB | ChromaDB (cosine similarity, /tmp persist) |
| Multi-agent | CrewAI (Layer 7) |
| State machine | LangGraph + MemorySaver (Layer 8) |
| Market data | yfinance |
| SEC filings | EDGAR API (free) |
| Fine-tuning | OpenAI fine-tune API |
| Hosting | Streamlit Community Cloud |

---

## 7. Production Deployment Considerations

**What would need to change for production:**

| Area | Current (prototype) | Production |
|------|-------------------|------------|
| Vector DB | ChromaDB on /tmp (resets hourly) | Pinecone or Weaviate managed |
| State persistence | LangGraph MemorySaver (in-memory) | PostgreSQL checkpointer |
| API keys | Streamlit Secrets | AWS Secrets Manager |
| Auth | None | OAuth2 / SSO |
| Cost control | $5/day soft cap | Per-advisor budget + alerts |
| Audit log | /tmp JSONL file | Append-only DB table |
| Scaling | Single Streamlit instance | FastAPI backend + Streamlit frontend |
| Observability | print() logs | Datadog / CloudWatch + tracing |

**Cost at scale (estimated for 50 advisors, 20 analyses/day):**
- Layer 7 crew analysis: ~$0.10/run × 1,000/day = $100/day
- Layer 8 rebalancing: ~$0.03/run × 200/day = $6/day
- Layer 9 committee: ~$0.08/run × 100/day = $8/day
- **Total: ~$114/day (~$3,400/month)** — well within enterprise budget for $12.8B AUM

---

## 8. Key Learnings

1. **Agents ≠ magic** — most "multi-agent" systems are sequential function calls with LLM personas. Real agency is when the LLM controls the flow (Layer 6).

2. **Frameworks add scaffolding, not intelligence** — CrewAI and LangGraph are optional. The same logic can be written in plain Python. They add retry logic, visualisation, and persistence.

3. **Token cost is the real constraint** — not model capability. Trimming prompts and capping output has more impact than choosing a cheaper model.

4. **RAG quality = chunking quality** — generic sliding windows produce poor retrieval. Section-aware chunking with per-section strategies (Layer 3) dramatically improves answer accuracy.

5. **Human-in-the-loop is a feature, not a limitation** — the L8 approval gate is the most valuable safety mechanism. Autonomous systems should earn trust incrementally.

---

## 9. Future Enhancements

- **L3**: Add earnings call transcript ingestion (audio → transcript → RAG)
- **L6**: Add broker API integration for real trade execution
- **L8**: Add tax-loss harvesting optimisation across multiple accounts
- **L9**: Add async committee debate (agents debate over time, not in one session)
- **L10**: Client portal (read-only view for clients to see their AI analysis)
- **All**: Replace OpenAI with Claude for cost reduction and longer context windows

---

## 10. Live Demo Flow (10-15 minutes)

| Time | Demo |
|------|------|
| 0:00 | Open app, show sidebar — 10 layers built |
| 1:00 | **L2**: Enter a portfolio, show live prices |
| 2:30 | **L3**: Ask a question about a 10-K filing |
| 4:00 | **L6**: Run ReAct agent, show tool-calling steps |
| 6:00 | **L7**: Run CrewAI crew, show 3-agent analysis |
| 8:30 | **L8**: Run rebalancing, trigger approval gate, approve |
| 11:00 | **L9**: Submit proposal, watch committee debate and vote |
| 13:00 | **L10**: Show architecture overview and cost log |
| 14:00 | Q&A |

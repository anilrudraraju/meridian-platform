# Meridian Intelligence Platform — Context for Builders

## What this is
A 10-layer AI wealth management platform built in Python + Streamlit, deployed at https://meridian-platform.streamlit.app. Single codebase: `app.py` (~2,300 lines) + `core/` modules. All layers are live and navigable via a sidebar.

## What the builder knows
Having built this end-to-end, the builder has hands-on experience with:
- **Prompt engineering**: zero-shot, few-shot, chain-of-thought, role-based, ReAct patterns
- **RAG pipelines**: document chunking strategies, ChromaDB vector search, cosine similarity, XBRL structured data retrieval, SEC EDGAR API
- **OpenAI API**: chat completions, tool/function calling, fine-tuning, embeddings (ada-002), token/cost tracking
- **Autonomous agents**: ReAct loop (think → tool call → observe → repeat), tool schemas, multi-step reasoning
- **Multi-agent systems**: CrewAI sequential crews, agent personas, task chaining, inter-agent context passing
- **Stateful workflows**: LangGraph StateGraph, conditional edges, human-in-the-loop interrupt/resume, MemorySaver checkpointing
- **Agent communication**: MessageBus pattern, multi-round debate protocols, consensus voting
- **Responsible AI**: PII detection, bias testing, guardrail validation, audit logging
- **Cost optimisation**: prompt compression, max_tokens caps, context truncation, model tiering

## Current app capabilities (what's already built)

| Layer | What it does | Key classes/functions |
|-------|-------------|----------------------|
| L1 | Prompt engineering demos | `FinancialPromptEngine`, `FinancialGuardrails` |
| L2 | Live portfolio valuation | `MarketDataFetcher` (yfinance) |
| L3 | SEC 10-K/10-Q document Q&A | `DocumentProcessor`, `RAGSystem`, `fetch_edgar_filing()`, `fetch_xbrl_facts()` |
| L4 | Base vs fine-tuned model comparison | `FinancialEvaluator`, ROUGE + semantic similarity |
| L5 | PII scanning, bias detection, audit log | `PIIScanner`, `BiasDetector`, `AuditLogger` |
| L6 | Autonomous portfolio monitoring agent | `PortfolioReActAgent`, `SafeAgentExecutor` |
| L7 | 3-agent analysis crew | `PortfolioAnalysisCrew` (CrewAI) |
| L8 | Portfolio rebalancing with human gate | `RebalancingWorkflow` (LangGraph) |
| L9 | Investment committee debate + vote | `InvestmentCommittee`, `MessageBus` |
| L10 | Integrated system overview | Architecture, cost log, production notes |

## Key APIs and interfaces

```python
# L3 — RAG
rag = RAGSystem()
rag.index_documents(chunks)
results = rag.search("What are the risk factors?", k=5)
answer = rag.answer_question("What was revenue in 2023?")

# L6 — ReAct Agent
agent = PortfolioReActAgent(model="gpt-4o")
result = agent.run("Monitor AAPL, MSFT portfolio", on_step=callback)
# result = {output, steps, iterations, cost_usd}

# L7 — CrewAI Crew
crew = PortfolioAnalysisCrew(model="gpt-4o")
result = crew.run({"AAPL": 0.40, "MSFT": 0.35, "GOOGL": 0.25}, on_task=callback)
# result = {output, agent_outputs, holdings}

# L8 — LangGraph Rebalancing
wf = RebalancingWorkflow()
result = wf.analyze(portfolio, target_allocation, approval_threshold=1_000_000)
# if result["status"] == "pending_approval": call wf.complete(approved=True/False)

# L9 — Investment Committee
committee = InvestmentCommittee(model="gpt-4o")
result = committee.run("Proposal text...", on_message=callback)
# result = {votes, tally, outcome: "APPROVED"/"REJECTED"/"MODIFICATION REQUIRED"}

# Cost tracking
from core.cost import log_call, daily_spend, read_log
```

## What can be built on top

**1. New data sources**
The RAG pipeline (`DocumentProcessor` + `RAGSystem`) accepts any text or PDF. You can feed earnings call transcripts, analyst reports, internal memos, or regulatory filings — not just SEC EDGAR.

**2. New agent tools**
`PortfolioReActAgent` uses a tool registry (`_TOOL_FUNCTIONS`, `_TOOL_SCHEMAS`). Add any function as a new tool (broker API, news feed, options data) and the agent will use it autonomously.

**3. New agent personas**
`InvestmentCommittee` is a list of dicts with `role`, `system`, `icon`. Add a 4th agent (e.g. ESG Specialist, Macro Economist) by appending to `_AGENTS`.

**4. New workflow nodes**
`RebalancingWorkflow` is a LangGraph StateGraph. Add nodes (e.g. compliance check, broker execution, client notification) and wire them with `add_edge` / `add_conditional_edges`.

**5. Different domains**
The architecture is domain-agnostic. The same pattern (RAG + ReAct agent + multi-agent crew + stateful workflow + committee vote) applies to legal document review, medical diagnosis support, insurance underwriting, or any advisory workflow.

**6. Production backend**
The Streamlit frontend can be separated from the `core/` modules. Any of the classes in `core/` can be wrapped in a FastAPI endpoint and called from a different frontend (React, mobile, etc.).

## Tech stack
Python 3.9+ · Streamlit · OpenAI API · ChromaDB · CrewAI · LangGraph · yfinance · SEC EDGAR API · pandas · sentence-transformers

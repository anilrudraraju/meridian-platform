# Meridian Intelligence Platform
### Global Fiscal Group — Capstone Project
**Course:** Generative AI and Agentic AI for Working Professionals

---

## What This Is
A live Streamlit web app implementing Layers 1–3 of the Meridian Intelligence Platform:
- **Layer 1:** Prompt Engineering + Guardrails (Tab 3)
- **Layer 2:** Real-Time Market Intelligence via yfinance (Tab 1)
- **Layer 3:** Document Intelligence + RAG via SEC EDGAR (Tab 2)

---

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open http://localhost:8501 in your browser
# 4. Enter your OpenAI API key in the sidebar
```

---

## Deploy to Streamlit Community Cloud (Free)

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your GitHub account
4. Select your repo → `app.py` as the main file
5. Click Deploy — your app gets a public URL instantly

No servers, no DevOps, completely free.

---

## Week 3 Assignment Deliverables Checklist

### Milestone 3.1 — Market Intelligence Layer
- [x] MarketDataFetcher using yfinance
- [x] Real-time portfolio valuation
- [x] Portfolio metrics (Day Change %, 1Y Return, Beta, Sector)
- [x] AI-powered commentary via GPT-4
- [x] Streamlit dashboard with live charts

### Milestone 3.2 — Document Intelligence Layer
- [x] EDGAR auto-fetch (10-K, 10-Q, 8-K) for any ticker
- [x] Manual PDF/TXT upload support
- [x] Text chunking (chunk_size=1000, overlap=200)
- [x] OpenAI text-embedding-ada-002 (1536 dimensions)
- [x] In-memory vector store with cosine similarity
- [x] Semantic search (similarity > 0.70, Top K=5)
- [x] Q&A with source citations
- [x] Q&A history export (JSON) for submission
- [ ] Run 25+ test questions across 3+ documents ← do this in the app

### Layer 1 — Guardrails
- [x] PII detection (SSN, credit card, email, phone, account numbers)
- [x] Prompt injection prevention
- [x] Prohibited topic filtering
- [x] Output compliance checking
- [x] Auto-disclaimer appending

---

## How to Meet the 25-Question Requirement

1. Go to Tab 2 (Document Intelligence)
2. Load 3+ documents (e.g. AAPL 10-K, MSFT 10-K, NVDA 10-K via EDGAR)
3. Build the vector index
4. Ask 25+ questions using the Q&A interface
5. Export the Q&A log (JSON) and include in your submission folder

---

## Future Weeks — What Gets Added

| Week | Layer | What's Added to This App |
|------|-------|--------------------------|
| 4 | Fine-Tuning | New tab: fine-tuned model vs base model comparison |
| 5 | Guardrails++ | Enhanced bias detection, hallucination detection |
| 6 | Agents | New tab: ReAct agent doing autonomous portfolio monitoring |
| 7-8 | Workflows | LangGraph workflow visualization |
| 9-10 | Multi-Agent | Investment committee debate UI |

---

## Submission Folder Structure
```
your_name_week3/
├── app.py                    ← this file
├── requirements.txt
├── README.md
├── results/
│   └── week3_qa_results.json ← exported from the app
└── docs/
    └── evaluation_report.md  ← your written analysis
```

---

## Tech Stack
- **Frontend:** Streamlit
- **LLM:** OpenAI GPT-4
- **Embeddings:** OpenAI text-embedding-ada-002
- **Market Data:** yfinance
- **Document Fetching:** SEC EDGAR API (free, no key needed)
- **Vector Search:** In-memory cosine similarity (upgrades to ChromaDB in Week 7)

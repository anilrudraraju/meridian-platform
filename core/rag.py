import os
import re
import io
import hashlib
from typing import List, Dict, Tuple, Optional

import streamlit as st

from core.dataclasses import SearchResult, RAGResponse
from core.constants import STRUCTURED_SIGNALS, NARRATIVE_SIGNALS
from core.chunking import get_chunking_config, _split_into_sections, _chunk_all_sections
from core.edgar import fetch_xbrl_facts, _fmt_xbrl

CHROMA_PERSIST_DIR = "/tmp/meridian_chromadb"
CHROMA_COLLECTION   = "meridian_docs"


class DocumentProcessor:
    """
    Process and chunk documents for RAG.
    Replicates RecursiveCharacterTextSplitter behavior from week3_capstone.ipynb.
    chunk_size=1000, chunk_overlap=200 (assignment spec)
    """

    def __init__(self, chunk_size=2000, chunk_overlap=400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str) -> List[Dict]:
        """Returns list of {page_content, metadata} dicts — matches LangChain Document format"""
        if not text or not text.strip():
            return []

        # Pre-scan for SEC Item headers so each chunk knows which section it's in.
        # Matches "ITEM 1.", "Item 1A.", "ITEM 7. MANAGEMENT'S DISCUSSION", etc.
        section_breaks = [
            (m.start(), f"Item {m.group(1)} — {m.group(2).strip()[:40]}")
            for m in re.finditer(r'(?m)^\s*(?:ITEM|Item)\s+(\d+[A-Z]?)\.?\s+([^\n]{0,60})', text)
        ]

        def current_section(pos: int) -> str:
            if not section_breaks:
                return ""
            label = "Preamble"
            for sec_pos, sec_label in section_breaks:
                if sec_pos <= pos:
                    label = sec_label
            return label

        chunks = []
        discarded = 0
        start = 0
        chunk_id = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end]
            if len(piece.strip()) >= 50:
                chunks.append({
                    "page_content": piece,
                    "metadata": {
                        "source": source,
                        "chunk_id": chunk_id,
                        "section": current_section(start),
                    }
                })
                chunk_id += 1
            else:
                discarded += 1
            start += step
        if discarded > 0:
            st.caption(f"ℹ️ {discarded} chunk(s) from '{source}' were too short (<50 chars) and skipped.")
        return chunks

    def load_from_text(self, text: str, source: str) -> List[Dict]:
        return self.chunk_text(text, source)

    def load_from_pdf_bytes(self, pdf_bytes: bytes, source: str,
                            table_aware: bool = False) -> List[Dict]:
        """
        PDF extraction with two modes:
        - Fast mode (default): pypdf — ~15-30s even for large 10-Qs.
        - Table-aware mode (opt-in): pdfplumber — preserves table structure but
          can take 2+ minutes on large documents (100+ pages).
        """
        text = ""
        used_pdfplumber = False

        # ── Table-aware mode: pdfplumber ──────────────────────────────────────
        if table_aware:
            try:
                import pdfplumber
                pages_text = []
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    for page in pdf.pages:
                        page_parts = []
                        try:
                            tables = page.extract_tables()
                            for table in tables:
                                for row in table:
                                    if row:
                                        clean_row = " | ".join(
                                            cell.strip() if cell else ""
                                            for cell in row
                                        )
                                        if clean_row.strip(" |"):
                                            page_parts.append(clean_row)
                        except Exception as e:
                            st.caption(f"⚠️ Table extraction skipped on a page of '{source}': {str(e)[:80]}")
                        try:
                            prose = page.extract_text()
                            if prose:
                                page_parts.append(prose)
                        except Exception as e:
                            st.caption(f"⚠️ Text extraction skipped on a page of '{source}': {str(e)[:80]}")
                        if page_parts:
                            pages_text.append("\n".join(page_parts))
                text = "\n".join(pages_text)
                used_pdfplumber = True
            except ImportError:
                st.warning("pdfplumber not installed — falling back to fast mode.")
            except Exception as e:
                st.warning(f"pdfplumber failed for '{source}': {str(e)[:100]}. Falling back to fast mode.")

        # ── Fast mode: pypdf (primary, or fallback from pdfplumber) ──────────
        if not text.strip():
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(pdf_bytes))
                pages_text = []
                for page_num, page in enumerate(reader.pages):
                    try:
                        extracted = page.extract_text()
                        if extracted:
                            pages_text.append(extracted)
                    except Exception as e:
                        st.caption(f"⚠️ pypdf skipped page {page_num + 1} of '{source}': {str(e)[:80]}")
                        continue
                text = "\n".join(pages_text)
            except Exception as e:
                st.warning(f"PDF extraction failed for '{source}': {str(e)[:100]}.")
                try:
                    text = pdf_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    text = ""

        if not text.strip():
            st.error(
                f"Could not extract text from {source}. "
                "The PDF may be scanned/image-based. Try a text-based PDF."
            )
            return []

        parser_used = "pdfplumber (table-aware)" if used_pdfplumber else "pypdf (fast)"
        st.caption(f"📄 Parsed with **{parser_used}** — {len(text):,} chars extracted")
        return self.chunk_text(text, source)

    def load_from_txt_bytes(self, txt_bytes: bytes, source: str) -> List[Dict]:
        return self.chunk_text(txt_bytes.decode("utf-8", errors="ignore"), source)

    def _extract_text_from_pdf(self, pdf_bytes: bytes, source: str) -> str:
        """Extract raw text from PDF bytes using pdfplumber (table-aware)."""
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    parts = []
                    try:
                        for row in (page.extract_tables() or []):
                            for r in row:
                                if r:
                                    line = " | ".join(c.strip() if c else "" for c in r)
                                    if line.strip(" |"):
                                        parts.append(line)
                    except Exception:
                        pass
                    try:
                        prose = page.extract_text()
                        if prose:
                            parts.append(prose)
                    except Exception:
                        pass
                    if parts:
                        pages.append("\n".join(parts))
            text = "\n".join(pages)
            if text.strip():
                st.caption(f"📄 Extracted with pdfplumber — {len(text):,} chars")
                return text
        except Exception:
            pass
        # Fallback: pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            text = "\n".join(
                p.extract_text() for p in reader.pages if p.extract_text()
            )
            if text.strip():
                st.caption(f"📄 Extracted with pypdf — {len(text):,} chars")
                return text
        except Exception as e:
            st.warning(f"PDF extraction failed for '{source}': {e}")
        return ""

    def process_filing(
        self,
        source:      str,
        company:     str,
        ticker:      str,
        cik:         str,
        form_type:   str,
        fiscal_year: str,
        quarter:     Optional[str] = None,
        period_end:  Optional[str] = None,
        pdf_bytes:   Optional[bytes] = None,
        text:        Optional[str] = None,
        text_source: str = "edgar_fetch",
    ) -> List[Dict]:
        """
        Unified entry point for all filing types and input formats.
        Provide either pdf_bytes (PDF upload) or text (EDGAR fetch), never both.
        Both paths produce identical chunk format and metadata schema.
        """
        if pdf_bytes is not None:
            raw_text = self._extract_text_from_pdf(pdf_bytes, source)
            text_source = "pdf_upload"
        elif text is not None:
            raw_text = text
        else:
            raise ValueError("Must provide either pdf_bytes or text")

        if not raw_text or not raw_text.strip():
            return []

        config = get_chunking_config(form_type)

        # For Form 10, try XBRL to determine if financials can be skipped
        xbrl_available = False
        if config["xbrl_expected"] and ticker:
            ok, _, _ = fetch_xbrl_facts(ticker)
            xbrl_available = ok

        # Build base metadata attached to every chunk
        base_meta = {
            "company": company or "", "ticker": ticker or "",
            "cik": cik or "", "form_type": form_type or "",
            "fiscal_year": fiscal_year or "", "quarter": quarter or "",
            "period_end": period_end or "", "filed_date": "",
            "source": source, "text_source": text_source,
            "audited": form_type == "10-K",
        }

        sections = _split_into_sections(raw_text, form_type)
        st.caption(f"📂 Detected sections: {', '.join(sections.keys())}")

        chunks = _chunk_all_sections(sections, config, base_meta, xbrl_available)

        # Sanitise metadata: ChromaDB requires str/int/float/bool — no None
        clean_chunks = []
        for chunk in chunks:
            meta = {k: ("" if v is None else (True if v is True else
                        (False if v is False else str(v) if not isinstance(v, (int, float)) else v)))
                    for k, v in chunk["metadata"].items()}
            chunk_key = f"{source}__{meta.get('section','')}__chunk_{meta.get('chunk_id',0)}"
            clean_chunks.append({
                "page_content": chunk["page_content"],
                "metadata": meta,
                "_id": hashlib.md5(chunk_key.encode()).hexdigest(),
            })
        return clean_chunks


class RAGSystem:
    """
    Complete RAG system for financial document intelligence.
    Source: week3_capstone.ipynb
    Vector store: ChromaDB PersistentClient (replaces Python list store)
    Embeddings: OpenAI text-embedding-ada-002 (1536 dims)
    Returns: SearchResult and RAGResponse dataclasses (unchanged from notebook)
    Confidence: High/Medium/Low based on avg cosine similarity (unchanged)
    """

    def __init__(self, model: str = "gpt-4o"):
        import openai, chromadb
        self.model = model
        self._openai = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        # PersistentClient writes a chroma.sqlite3 file to CHROMA_PERSIST_DIR.
        # On Streamlit Cloud this persists within a single deployment session.
        # Locally it persists indefinitely across runs.
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._collection = self._chroma.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}   # native cosine similarity
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @property
    def _indexed(self) -> bool:
        return self._collection.count() > 0

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        """Delete and recreate the collection (wipe all documents)."""
        self._chroma.delete_collection(CHROMA_COLLECTION)
        self._collection = self._chroma.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Call OpenAI embedding API in batches.
        Truncates each text to 6000 chars (~1500 tokens) to stay within
        ada-002 limit of 8191 tokens per item. 1000-char chunks are ~250 tokens so
        this only triggers on unusually long EDGAR lines.
        batch_size=100 keeps total request size well under API limits while
        reducing network round-trips vs. the old batch_size=20.
        """
        # Warn before truncation — 5800 leaves margin before the 6000-char hard limit
        for t in texts:
            if len(t) > 5800:
                st.caption(f"⚠️ Chunk truncated from {len(t):,} to 6,000 chars before embedding.")
        safe_texts = [t[:6000] if len(t) > 6000 else t for t in texts]
        all_embeddings = []
        for i in range(0, len(safe_texts), batch_size):
            batch_num = i // batch_size + 1
            try:
                resp = self._openai.embeddings.create(
                    model="text-embedding-ada-002",
                    input=safe_texts[i:i + batch_size]
                )
                all_embeddings.extend([item.embedding for item in resp.data])
            except Exception as e:
                st.error(f"Embedding error on batch {batch_num}: {e}")
                raise
        return all_embeddings

    # ── core methods (matching week3_capstone.ipynb public interface) ─────────

    def index_documents(self, chunks: List[Dict]) -> None:
        """
        Embed chunks and upsert into ChromaDB.
        chunks: list of {page_content, metadata} dicts  (DocumentProcessor output)
        Uses upsert so re-indexing the same document overwrites existing chunks.
        """
        texts     = [c["page_content"] for c in chunks]
        metadatas = [c["metadata"]     for c in chunks]

        # Warn if any source being indexed already exists in the collection
        if self._collection.count() > 0:
            incoming_sources = {m.get("source", "") for m in metadatas}
            existing = self._collection.get(include=["metadatas"])
            existing_sources = {m.get("source", "") for m in existing["metadatas"]}
            overlap = incoming_sources & existing_sources
            if overlap:
                st.warning(f"⚠️ Re-indexing existing document(s): {', '.join(overlap)}. Previous chunks will be overwritten.")

        # Build stable IDs — use pre-computed _id from process_filing() when present
        # (includes section in the key so chunks from different sections don't collide).
        # Fall back to source+chunk_id for chunks from the legacy load_from_* path.
        ids = [
            c.get("_id") or hashlib.md5(
                f"{m.get('source','doc')}__{m.get('section','')}_chunk_{m.get('chunk_id', i)}".encode()
            ).hexdigest()
            for i, (c, m) in enumerate(zip(chunks, metadatas))
        ]
        # ChromaDB metadata values must be str/int/float/bool
        safe_meta = [
            {k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
             for k, v in m.items()}
            for m in metadatas
        ]

        progress = st.progress(0, text=f"Creating embeddings for {len(texts)} chunks (text-embedding-ada-002)...")
        all_embeddings = self._embed(texts)
        progress.progress(1.0, text=f"Embeddings complete — {len(texts)} chunks embedded.")

        # Upsert in batches of 100 — avoids ChromaDB internal size limits
        upsert_batch = 100
        for i in range(0, len(texts), upsert_batch):
            self._collection.upsert(
                documents=texts[i:i+upsert_batch],
                embeddings=all_embeddings[i:i+upsert_batch],
                metadatas=safe_meta[i:i+upsert_batch],
                ids=ids[i:i+upsert_batch]
            )
        progress.empty()

    def search(self, query: str, k: int = 20, where: Optional[dict] = None) -> List[SearchResult]:
        """
        Semantic search via ChromaDB.
        Returns List[SearchResult] — identical interface to week3_capstone.ipynb.
        ChromaDB cosine distance ∈ [0,2]; we convert to similarity ∈ [0,1].
        where: optional ChromaDB metadata filter (build with build_chroma_filter())
        """
        if not self._indexed:
            raise ValueError("No documents indexed. Call index_documents first.")

        q_emb = self._embed([query])[0]
        query_kwargs: dict = dict(
            query_embeddings=[q_emb],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        if where:
            query_kwargs["where"] = where
        results = self._collection.query(**query_kwargs)

        search_results = []
        docs      = results.get("documents", [[]])[0] if results.get("documents") else []
        metas     = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []

        for doc, meta, dist in zip(docs, metas, distances):
            # Convert ChromaDB cosine distance → similarity score (0–1, higher = better)
            similarity = max(0.0, 1.0 - (dist / 2.0))
            search_results.append(SearchResult(
                content=doc,
                source=meta.get("source", "Unknown"),
                relevance_score=similarity,
                metadata=meta
            ))
        return search_results

    def answer_question(self, question: str, k: int = 20) -> RAGResponse:
        """
        RAG Q&A — returns RAGResponse matching week3_capstone.ipynb exactly.
        Confidence: High (>0.80) / Medium (>0.70) / Low based on avg similarity.
        """
        results = self.search(question, k=k)
        # Drop chunks that are too dissimilar to be useful — they add noise to the context
        results = [r for r in results if r.relevance_score >= 0.50]
        if not results:
            return RAGResponse(
                question=question,
                answer="I don't have enough information to answer this question — no relevant chunks were found in the index.",
                sources=[],
                confidence="Low"
            )

        context = "\n\n".join([
            f"[Source {i+1}]\n{r.content}" for i, r in enumerate(results)
        ])
        prompt = f"""You are a financial analyst. Answer the question based ONLY on the provided context.

IMPORTANT RULES:
1. If the answer is not in the context, say "I don't have enough information."
2. Cite sources using [Source X] notation
3. Do not add information not present in the context
4. Be specific and factual

Context:
{context}

Question: {question}

Answer (with source citations):"""

        resp = self._openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = resp.choices[0].message.content
        avg_score = sum(r.relevance_score for r in results) / len(results)
        confidence = "High" if avg_score > 0.75 else "Medium" if avg_score > 0.60 else "Low"
        return RAGResponse(question=question, answer=answer, sources=results, confidence=confidence)

    def answer_with_context(self, question: str, context: str) -> str:
        """Generate an answer from pre-built context (used by the dual-store retrieve path)."""
        prompt = f"""You are a financial analyst at Meridian Wealth Partners.
Answer the question using ONLY the data provided below.
If the answer is not in the data, say "I don't have enough information."
Be specific and cite figures directly.

{context}

Question: {question}

Answer:"""
        resp = self._openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content

    def analyze_risk_factors(self, company: str) -> RAGResponse:
        """week3_capstone.ipynb method — unchanged"""
        return self.answer_question(f"What are the main risk factors for {company}?")

    def summarize_earnings(self, company: str, quarter: str) -> RAGResponse:
        """week3_capstone.ipynb method — unchanged"""
        return self.answer_question(f"Summarize the key points from {company}'s {quarter} earnings call")


# ── Query routing & retrieval ─────────────────────────────────────────────────

def build_chroma_filter(filters: dict) -> Optional[dict]:
    """Build a ChromaDB $and filter from a dict, dropping None and empty values."""
    valid = {k: str(v) for k, v in filters.items() if v is not None and v != ""}
    if not valid:
        return None
    if len(valid) == 1:
        k, v = next(iter(valid.items()))
        return {k: {"$eq": v}}
    return {"$and": [{k: {"$eq": v}} for k, v in valid.items()]}


def route_query(question: str) -> str:
    """Classify question as 'structured', 'narrative', or 'both'."""
    q = question.lower()
    s = sum(1 for sig in STRUCTURED_SIGNALS if sig in q)
    n = sum(1 for sig in NARRATIVE_SIGNALS if sig in q)
    if s > n: return "structured"
    if n > s: return "narrative"
    return "both"


def retrieve(question: str, ticker: str, fiscal_year: str, form_type: str,
             quarter: Optional[str], rag: "RAGSystem",
             xbrl_facts: dict) -> str:
    """
    Route question → pull XBRL facts and/or RAG chunks → return combined context string.
    """
    store = route_query(question)
    q_lower = question.lower()
    parts = []

    # ── Structured path: XBRL metrics ────────────────────────────────────────
    if store in ("structured", "both") and xbrl_facts:
        metric_lines = []
        for label, entries in xbrl_facts.items():
            period_entries = [e for e in entries if e.get("form") == form_type] if form_type else entries
            if fiscal_year:
                period_entries = [e for e in period_entries
                                  if e.get("period_end", "").startswith(fiscal_year)]
            if period_entries:
                e = period_entries[0]
                metric_lines.append(
                    f"{label} ({e.get('period_end', '')}): {_fmt_xbrl(e['value'], label)}"
                )
        if metric_lines:
            parts.append("STRUCTURED DATA (XBRL):\n" + "\n".join(metric_lines))

    # ── Narrative path: RAG chunks ────────────────────────────────────────────
    if store in ("narrative", "both") and rag and rag.count() > 0:
        # Detect section hint from question keywords
        section_hint = None
        if any(w in q_lower for w in ("risk", "risks")):
            section_hint = "risk_factors"
        elif any(w in q_lower for w in ("strategy", "business", "competition")):
            section_hint = "business"
        elif any(w in q_lower for w in ("why", "margin", "revenue", "md&a", "management")):
            section_hint = "mdna"
        elif any(w in q_lower for w in ("note", "debt", "lease", "footnote")):
            section_hint = "footnotes"

        where = build_chroma_filter({
            "ticker":      ticker,
            "fiscal_year": fiscal_year,
            "form_type":   form_type,
            "quarter":     quarter,
            "section":     section_hint,
        })
        results = rag.search(question, k=20, where=where)
        results = [r for r in results if r.relevance_score >= 0.50]
        if results:
            narrative = "\n\n".join(
                f"[Source {i+1} | {r.metadata.get('section','')} "
                f"| score {r.relevance_score:.2f}]\n{r.content}"
                for i, r in enumerate(results)
            )
            parts.append("NARRATIVE CONTEXT (RAG):\n" + narrative)

    return "\n\n---\n\n".join(parts) if parts else ""


# Only used on the XBRL path where context is programmatically built and may be empty.
# NOT applied to the RAG path — when chunks are found GPT's answer is always shown as-is.
_INSUFFICIENT_PATTERNS = [
    "i don't have enough information",
    "don't have enough information",
    "not enough information to answer",
    "cannot calculate",
    "unable to calculate",
    "unable to answer",
    "i don't have the data",
]

def _gpt_refused(answer: str) -> bool:
    """Return True when GPT signals it cannot answer due to missing data (XBRL path only)."""
    a = answer.lower()
    return any(p in a for p in _INSUFFICIENT_PATTERNS)

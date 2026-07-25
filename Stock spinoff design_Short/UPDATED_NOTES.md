# Design Updates — Article/Newsletter Support

## What Changed

Added support for **Article/Newsletter documents** as a 4th source type (in addition to EDGAR, PDF, Manual).

## Document Source Types

```javascript
source: "EDGAR" | "PDF" | "Article" | "Manual"
```

### Source Details

| Type | Input | Processing | Use Case |
|------|-------|-----------|----------|
| **EDGAR** | Ticker + filing type | Auto-fetch from SEC, parse with LLM | Official SEC filings (10-K, 10-Q, 8-K) |
| **PDF** | File upload | OCR + LLM extraction | Investor presentations, internal docs |
| **Article** | URL paste | Fetch + summarize with Claude | Newsletters, research reports, blog posts, news |
| **Manual** | Free-text input | User-typed notes | Quick observations, call notes |

## API Endpoint

```
POST /api/workspace/:ticker/documents
Body: {
  source: "EDGAR" | "PDF" | "Article" | "Manual",
  content: string,           // URL for Article, text for Manual, file path for PDF
  activates: ["S1", "B2"],   // Which criteria does this address?
  title: string              // User-provided or auto-extracted
}
Response: {
  document: {
    id, title, source, date, activates, state
  },
  activatedCriteria: [...]
}
```

## UI Flow

**"+ Add Document" modal** → Radio buttons or tabs:
```
📊 EDGAR filing  |  📄 Upload PDF  |  🔗 Paste URL  |  ✍️ Manual notes
```

**For Article (URL):**
1. User pastes URL
2. Backend fetches + summarizes (Claude)
3. Shows preview: title, key quotes, auto-suggested criteria
4. User confirms criteria checkboxes
5. Save as Document with `source: "Article"`

## Data Model Update

```javascript
documents: [
  {
    id: "doc1",
    title: "Form 10 §9.2 — Pension liabilities",
    source: "EDGAR",
    date: "2026-05-14",
    activates: ["C2", "C5"],
    state: "ingested"
  },
  {
    id: "doc8",
    title: "Bernstein: LUMN Forced-Selling Spinoff Deep Dive",
    source: "Article",        // NEW
    date: "2026-05-20",
    activates: ["S1", "B2", "V3"],
    state: "ingested"
  }
  // ... rest of documents
]
```

## Integration Checklist

- [ ] Add "Article" as a source type in the document model
- [ ] Build UI for URL input in "+ Add Document" modal
- [ ] Implement backend: fetch URL → extract summary → suggest criteria
- [ ] Auto-index article text for corpus Q&A
- [ ] Show source icon in document list (Article = 🔗)
- [ ] Test: paste newsletter URL → criteria auto-suggest → save

## Notes for Claude Code

- **Newsletter URLs:** Can be any public article; backend should handle both newsletter platforms and direct URLs
- **Auto-suggestion:** Use Claude to read article + suggest which criteria it addresses (user can override)
- **Searchability:** Article content should be indexed in the corpus so Q&A can find relevant quotes
- **Attribution:** Keep the URL for audit trail ("Bernstein analysis from [link]")

---

**Files to update in your code:**
1. Add "Article" to `source` enum
2. Add URL fetch + summarization in document ingestion handler
3. Update "+ Add Document" UI to include Article tab
4. Index article text in corpus search

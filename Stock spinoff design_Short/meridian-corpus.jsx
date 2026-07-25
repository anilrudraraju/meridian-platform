/* global React, SpinoffUI, MeridianData */
const { I: CIc } = SpinoffUI;
const { DIMS: CDIMS, DOCS: CDOCS, CRITERIA: CCRIT } = MeridianData;

/* ====================================================================
   Ask the corpus — grounded Q&A panel
   Retrieval is always against ALL workspace docs.
   Each question carries an editable dimension tag (auto-detected,
   user can override). The tag drives pin-to-criterion suggestions
   and grouping in the Saved Q&A drawer.
==================================================================== */

/* Simple keyword → dimension classifier (stand-in for an LLM tagger). */
const DIM_KEYWORDS = {
  setup:      ["index", "forced selling", "forced-selling", "spin", "distribution", "ratio", "float", "institutional"],
  business:   ["roic", "fcf", "free cash flow", "margin", "growth", "customer", "moat", "segment", "concentration", "operating leverage"],
  capital:    ["debt", "pension", "leverage", "ebitda", "maturity", "capex", "covenant", "interest coverage", "off-balance", "opeb"],
  valuation:  ["ev/", "ebit", "p/fcf", "multiple", "sotp", "sum-of-parts", "sum of parts", "peer", "valuation", "yield"],
  incentives: ["comp", "ownership", "buyback", "insider", "option", "vesting", "stake", "form 4", "def 14a"],
};
function classifyQuestion(q) {
  if (!q || q.length < 3) return null;
  const t = q.toLowerCase();
  const scores = {};
  for (const [dim, kws] of Object.entries(DIM_KEYWORDS)) {
    scores[dim] = kws.reduce((n, k) => n + (t.includes(k) ? 1 : 0), 0);
  }
  const best = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
  return best[1] > 0 ? best[0] : null;
}

function dimMeta(id) { return CDIMS.find(d => d.id === id); }

/* Pin-suggestion lookup — pick a sensible open criterion in a dim */
function pinForDim(dimId) {
  if (!dimId) return null;
  const crits = CCRIT[dimId] || [];
  const c = crits.find(x => x.status === "blocked")
        || crits.find(x => x.status === "open")
        || crits.find(x => x.status === "partial")
        || crits[0];
  return c ? { dim: dimId, crit: c.name } : null;
}

/* ----- Seed thread (pre-populated demo) — chronological order
   so the render-time reverse() yields newest-first. ----- */
const SEED_THREAD = [
  { role: "user", q: "Is customer concentration ever quantified — top 10 customers as a % of revenue?", dim: "business", time: "8m ago" },
  { role: "assistant",
    confidence: "high",
    confidenceWhy: "Direct disclosure in Form 10 with a single, unambiguous figure.",
    spans: [
      { t: "Yes — Form 10 §3.2 discloses that " },
      { t: "the top 10 customers represented 31% of FY25 revenue", cite: 1 },
      { t: ", with " },
      { t: "no single customer above 6%", cite: 1 },
      { t: ". This is below the 40%+ threshold that typically triggers a Business Quality penalty for concentration risk." },
    ],
    cites: [
      { n: 1, doc: "Form 10 · §3.2 · p.34", snippet: "The Company's ten largest customers in fiscal year 2025 accounted for approximately 31% of net revenue, with no single customer accounting for more than approximately 6% of net revenue." },
    ],
    saved: false,
  },
  { role: "user", q: "What did mgmt say about the pension funding shortfall, and is the size disclosed anywhere?", dim: "capital", time: "2m ago" },
  { role: "assistant",
    confidence: "medium",
    confidenceWhy: "CFO statements are clear but the funding number itself is not in the corpus yet — the May 23 footnote is still pending.",
    spans: [
      { t: "Pension obligation of " },
      { t: "$340M is disclosed in Form 10", cite: 1 },
      { t: ", but the funding shortfall — i.e. how much of that is unfunded — has not been quantified. On the Q1 call, " },
      { t: "the CFO declined twice to give a number", cite: 2 },
      { t: ", saying the actuarial review will land with the spinoff registration footnote. " },
      { t: "Sell-side consensus implies ~$90M of additional contributions over 3 years", cite: 3 },
      { t: ", but this is not corroborated by management. The pension footnote was expected today (May 23) and would close this gap." },
    ],
    cites: [
      { n: 1, doc: "Form 10 · p.87",        snippet: "…the Company assumes a projected benefit obligation of $340 million as of December 31, 2025. The funded status is described as 'substantially funded' but the unfunded amount is not separately disclosed…" },
      { n: 2, doc: "Q1 2026 transcript · p.14", snippet: "Analyst (Goldman): Can you size the pension contribution we should model? · CFO: We've consistently said we'll provide the actuarial detail in the spinoff registration footnote and I don't want to get ahead of that work." },
      { n: 3, doc: "Sell-side note · Goldman · May 7", snippet: "Our model implies ~$90M of incremental pension contributions across FY27–FY29 vs. the parent's run-rate, based on PBGC peer comps." },
    ],
    saved: true,
  },
];

const SUGGESTIONS = [
  { q: "Are there any operating segment changes versus the parent's reporting?", dim: "business" },
  { q: "What's the debt maturity wall in 2028 — is it refinanceable?", dim: "capital" },
  { q: "Has insider buying happened post-Form 10 filing?", dim: "incentives" },
  { q: "What forced-selling estimate does the corpus support?", dim: "setup" },
];

/* ----- Components ----- */

function ConfidencePip({ level }) {
  const map = {
    high:   { c: "var(--green)",  l: "High confidence",   pips: 3 },
    medium: { c: "var(--amber)",  l: "Medium confidence", pips: 2 },
    low:    { c: "var(--red)",    l: "Low confidence",    pips: 1 },
  };
  const m = map[level] || map.medium;
  return (
    <span className="m-ask-conf" title={m.l}>
      {[1,2,3].map(n => (
        <span key={n} className="b" style={{ background: n <= m.pips ? m.c : "rgba(255,255,255,0.08)" }} />
      ))}
      <span className="lbl">{m.l}</span>
    </span>
  );
}

/* Editable dim chip — used on questions in the thread AND below the composer.
   Click to open a small popover; pick a different dim or "untag". */
function DimTagChip({ value, onChange, source, size = "md" }) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);
  React.useEffect(() => {
    if (!open) return;
    const close = (e) => { if (!ref.current?.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const d = value ? dimMeta(value) : null;
  return (
    <span className={"m-ask-dimtag " + (size === "sm" ? "sm " : "") + (d ? d.cls : "untagged")} ref={ref}>
      <button className="chip" onClick={() => setOpen(v => !v)}>
        {d ? (
          <>
            <span className="letter">{d.letter}</span>
            <span className="name">{d.name}</span>
          </>
        ) : (
          <>
            <span className="letter q">?</span>
            <span className="name">Untagged</span>
          </>
        )}
        {source === "auto" && <span className="src" title="Auto-detected by Meridian">auto</span>}
        <span className="caret">▾</span>
      </button>
      {open && (
        <div className="m-ask-dimtag-menu">
          <div className="h">Tag this question</div>
          {CDIMS.map(dim => (
            <button key={dim.id}
              className={"opt " + dim.cls + (value === dim.id ? " on" : "")}
              onClick={() => { onChange(dim.id); setOpen(false); }}>
              <span className="letter">{dim.letter}</span>
              <span className="name">{dim.name}</span>
              {value === dim.id && <span className="check">{CIc.check}</span>}
            </button>
          ))}
          <button className={"opt untagged " + (value === null ? "on" : "")}
            onClick={() => { onChange(null); setOpen(false); }}>
            <span className="letter q">?</span>
            <span className="name">Untagged</span>
          </button>
        </div>
      )}
    </span>
  );
}

function CiteChip({ c, active, onToggle }) {
  return (
    <button className={"m-ask-cite-chip " + (active ? "active" : "")} onClick={onToggle}>
      <span className="n">[{c.n}]</span>
      <span className="d">{c.doc}</span>
    </button>
  );
}

function PinPopover({ suggestion, onClose }) {
  if (!suggestion) {
    return (
      <div className="m-ask-pin-pop" onClick={(e) => e.stopPropagation()}>
        <div className="h">Pin answer to criterion</div>
        <div className="muted" style={{ padding: "12px 14px", fontSize: 12, lineHeight: 1.5 }}>
          Tag this question with a dimension first — the pin target depends on which bucket the answer belongs to.
        </div>
        <div className="actions">
          <button className="btn ghost" style={{ padding: "5px 10px", fontSize: 12 }} onClick={onClose}>OK</button>
        </div>
      </div>
    );
  }
  const d = dimMeta(suggestion.dim);
  return (
    <div className="m-ask-pin-pop" onClick={(e) => e.stopPropagation()}>
      <div className="h">Pin answer to criterion</div>
      <div className="row">
        <span className={"letter " + d.cls}>{d.letter}</span>
        <div>
          <div className="dim">{d.name}</div>
          <div className="crit">{suggestion.crit}</div>
        </div>
      </div>
      <div className="muted" style={{ fontSize: 11.5, padding: "0 12px 8px", lineHeight: 1.4 }}>
        Adds this answer + citations to the criterion value, with provenance.
      </div>
      <div className="actions">
        <button className="btn ghost" style={{ padding: "5px 10px", fontSize: 12 }} onClick={onClose}>Cancel</button>
        <button className="btn pri" style={{ padding: "5px 10px", fontSize: 12 }} onClick={onClose}>Pin {CIc.arrowR}</button>
      </div>
    </div>
  );
}

function AnswerBlock({ msg, questionDim, onTraceToggle, openCite }) {
  const [pinOpen, setPinOpen] = React.useState(false);
  const [saved, setSaved] = React.useState(msg.saved);
  const pinSuggestion = pinForDim(questionDim);
  const pinDim = pinSuggestion ? dimMeta(pinSuggestion.dim) : null;
  return (
    <div className="m-ask-row machine">
      <span className="ava">M</span>
      <div className="bubble">
        <div className="conf-line">
          <ConfidencePip level={msg.confidence} />
          <span className="why">{msg.confidenceWhy}</span>
        </div>
        <div className="body">
          {msg.spans.map((s, i) => s.cite
            ? <span key={i}>{s.t}<sup className={"cite-marker " + (openCite === s.cite ? "active" : "")}
                onClick={() => onTraceToggle(s.cite)}>[{s.cite}]</sup></span>
            : <span key={i}>{s.t}</span>
          )}
        </div>
        <div className="cites">
          {msg.cites.map(c => (
            <CiteChip key={c.n} c={c} active={openCite === c.n} onToggle={() => onTraceToggle(c.n)} />
          ))}
        </div>
        {openCite && (() => {
          const c = msg.cites.find(x => x.n === openCite);
          return c ? (
            <div className="snippet">
              <div className="snippet-head">
                <span className="n">[{c.n}]</span>
                <span className="d">{c.doc}</span>
                <button className="btn ghost" style={{ marginLeft: "auto", padding: "3px 8px", fontSize: 11 }}>Open document {CIc.arrowUR}</button>
              </div>
              <div className="snippet-body">"{c.snippet}"</div>
            </div>
          ) : null;
        })()}
        <div className="actions">
          <button className={"btn " + (saved ? "saved" : "")} onClick={() => setSaved(v => !v)}>
            {CIc.bookmark}{saved ? "Saved to notes" : "Save to notes"}
          </button>
          <div style={{ position: "relative" }}>
            <button className="btn" onClick={() => setPinOpen(v => !v)}>
              {pinDim
                ? <><span className={"dim-dot " + pinDim.cls} /> Pin to {pinDim.name}</>
                : <>Pin to criterion…</>}
            </button>
            {pinOpen && <PinPopover suggestion={pinSuggestion} onClose={() => setPinOpen(false)} />}
          </div>
          <button className="btn">{CIc.tag} Tag</button>
          <button className="btn ghost">Disagree</button>
        </div>
      </div>
    </div>
  );
}

function QuestionBlock({ msg, onDimChange }) {
  return (
    <div className="m-ask-row user">
      <span className="ava user">{CIc.user}</span>
      <div className="bubble">
        <div className="q">{msg.q}</div>
        <div className="qmeta">
          <DimTagChip value={msg.dim || null} onChange={onDimChange} source={msg.dimSource} size="sm" />
          <span className="t">{msg.time}</span>
        </div>
      </div>
    </div>
  );
}

function AskCorpusPanel({ onClose }) {
  const [q, setQ] = React.useState("");
  const [tag, setTag] = React.useState(null);
  const [tagSource, setTagSource] = React.useState("auto");
  const [thread, setThread] = React.useState(SEED_THREAD);
  const [pending, setPending] = React.useState(false);
  const [openCiteByMsg, setOpenCite] = React.useState({});

  const indexedDocs = CDOCS.filter(d => d.state === "indexed").length;

  React.useEffect(() => {
    if (tagSource === "user") return;
    setTag(classifyQuestion(q));
  }, [q]); // eslint-disable-line

  const submit = () => {
    if (!q.trim()) return;
    const userMsg = { role: "user", q: q.trim(), dim: tag, dimSource: tagSource, time: "just now" };
    setThread(t => [...t, userMsg]);
    setQ("");
    setTag(null);
    setTagSource("auto");
    setPending(true);
    setTimeout(() => setPending(false), 1200);
  };

  const updateMsgDim = (idx, newDim) => {
    setThread(t => t.map((m, i) => i === idx ? { ...m, dim: newDim, dimSource: "user" } : m));
  };

  /* Group flat thread into Q→A pairs, then reverse for newest-first */
  const pairs = [];
  for (let i = 0; i < thread.length; i += 2) {
    pairs.push({ qIdx: i, aIdx: i + 1 });
  }
  const orderedPairs = [...pairs].reverse();

  return (
    <div className="m-ask-panel">
      {/* ---------- Composer pinned at top ---------- */}
      <div className="m-ask-composer">
        <span className="search-ico">{CIc.search}</span>
        <input
          className="m-ask-input-line"
          placeholder="Ask the corpus — anything across all documents"
          value={q}
          onChange={e => setQ(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
          autoFocus />
        <DimTagChip
          value={tag}
          source={tagSource}
          onChange={(v) => { setTag(v); setTagSource("user"); }} />
        <button className="btn pri ask-send" onClick={submit} disabled={!q.trim()}>
          Ask {CIc.arrowR}
        </button>
        <button className="m-ask-close" onClick={onClose} title="Close">×</button>
      </div>

      {/* ---------- Sub-bar ---------- */}
      <div className="m-ask-subbar">
        <span className="left">
          <span className="dot" />
          Grounded across <b>all {indexedDocs} documents</b> · 3,201 chunks · BM25 + embeddings
        </span>
        <span className="right">
          {q.trim().length > 2 ? (
            tag
              ? (tagSource === "auto"
                  ? <>Auto-tagged as <b>{dimMeta(tag).name}</b> — click the tag to change</>
                  : <>Tagged as <b>{dimMeta(tag).name}</b></>)
              : <>Couldn't auto-tag — leave untagged or pick a dimension</>
          ) : (
            <>Press <kbd>Enter</kbd> to ask · newest answers appear at top</>
          )}
        </span>
      </div>

      {/* ---------- Suggestions (always visible when no draft is being typed) ---------- */}
      {q.trim() === "" && (
        <div className="m-ask-sug-row">
          <span className="lbl">Try</span>
          {SUGGESTIONS.map((s, i) => {
            const d = dimMeta(s.dim);
            return (
              <button key={i} className={"sug-chip " + d.cls} onClick={() => { setQ(s.q); setTag(s.dim); setTagSource("auto"); }}>
                <span className="letter">{d.letter}</span>
                <span className="text">{s.q}</span>
              </button>
            );
          })}
        </div>
      )}

      {/* ---------- Thread (descending — newest first) ---------- */}
      <div className="thread">
        {pending && (
          <div className="m-ask-row machine">
            <span className="ava">M</span>
            <div className="bubble pending">
              <div className="pulse-row">
                <span className="step">{CIc.search} Retrieving · BM25 + embeddings · top-24 chunks</span>
                <span className="step">{CIc.copy2} Reranking · cross-encoder · top-6</span>
                <span className="step current"><span className="dot" />Composing with Sonnet · citing every claim…</span>
              </div>
            </div>
          </div>
        )}

        {orderedPairs.length === 0 && !pending && (
          <div className="m-ask-empty-inline">
            No questions yet — type one above. Meridian auto-tags the question with a dimension; change it any time.
          </div>
        )}

        {orderedPairs.map(pair => {
          const qMsg = thread[pair.qIdx];
          const aMsg = thread[pair.aIdx];
          return (
            <div key={pair.qIdx} className="m-ask-pair">
              <QuestionBlock msg={qMsg} onDimChange={(v) => updateMsgDim(pair.qIdx, v)} />
              {aMsg && (
                <AnswerBlock msg={aMsg}
                  questionDim={qMsg.dim || null}
                  openCite={openCiteByMsg[pair.aIdx]}
                  onTraceToggle={(n) => setOpenCite(s => ({ ...s, [pair.aIdx]: s[pair.aIdx] === n ? null : n }))} />
              )}
            </div>
          );
        })}
      </div>

      {/* ---------- Foot ---------- */}
      <div className="m-ask-foot">
        <span className="muted">Grounded mode · Meridian refuses to answer outside the corpus</span>
        <button className="btn ghost">{CIc.bookmark} 6 saved answers</button>
      </div>
    </div>
  );
}

window.MeridianCorpus = { AskCorpusPanel };

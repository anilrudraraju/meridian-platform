/* global React, SpinoffUI, MeridianData */
/* =====================================================================
   meridian-intake.jsx — New Analysis intake (multi-step) +
                         DocumentReviewPanel (the diff/preview surface) +
                         ConflictRow (side-by-side evidence picker)

   Design rationale
   ----------------
   The 5-bucket workspace is the brain. Two interactions deserve the
   most care because everything else hangs off them:

     1.  How a new analysis BEGINS — the user types a ticker; we should
         already be doing useful work before they finish breathing.
         The intake validates on EDGAR, auto-fetches Form 10, and runs
         an initial extraction pass across all 5 buckets — then HANDS
         THE CONTROL BACK to the user via a diff/preview. Nothing is
         written until the user confirms (per PRD).

     2.  How a NEW DOCUMENT enters an existing workspace — same diff
         surface, but now with possible CONFLICTS against existing
         values. We surface conflicts as a side-by-side evidence
         picker; user picks A, picks B, keeps both as flagged, or
         skips. Never auto-resolved.

   We use scores (not %) and the threshold is a SOFT NUDGE.

   Reusable surface: DocumentReviewPanel is exported standalone so
   it can be reopened from the workspace "Add document" action.
   ===================================================================== */

const { I: II } = SpinoffUI;
const { DIMS: DIMS_INTAKE } = MeridianData;

/* =====================================================================
   SAMPLE DATA — what an initial Form 10 scan proposes for LUMN
   In production, this comes from the Haiku extractor + XBRL loader.
   ===================================================================== */

const SAMPLE_FORM10_PROPOSALS = {
  doc: {
    name: "Form 10",
    type: "10-12B",
    pages: 312,
    source: "EDGAR · CIK 0000018926",
    fetched: "1.4 min ago",
    accession: "0000018926-26-000041",
    filedOn: "Mar 15, 2026",
  },
  byDim: {
    setup: {
      proposed: [
        { name: "Market cap",                value: "$1.6B",                    cite: "p.12 — \"Market capitalization at distribution is estimated at $1.6 billion…\"", conf: "High" },
        { name: "Index exclusion",           value: "3 indices",                cite: "p.12 — S&P MidCap 400 · Russell 1000 · MSCI USA Mid Cap",                       conf: "High" },
        { name: "Free float at spin",        value: "94%",                      cite: "p.18 — distribution mechanics",                                                  conf: "High" },
        { name: "Forced selling estimate",   value: "~$410M",                   cite: "p.18 + peer dataset",                                                            conf: "Med",  note: "Estimate combines float × index-fund holdings" },
        { name: "Strategic rationale",       value: "Articulated",              cite: "§1 — strategic separation rationale",                                            conf: "High" },
        { name: "Spin dates & ratio",        value: "Jun 14, 2026 · 1:4",       cite: "p.6 — distribution terms",                                                       conf: "High" },
      ],
      unanswered: [
        { name: "Institutional ownership",   reason: "Needs latest 13F" },
      ],
    },
    business: {
      proposed: [
        { name: "ROIC",                       value: "18.4%",                   cite: "XBRL · FY25",                                                                     conf: "High" },
        { name: "FCF conversion",             value: "92%",                     cite: "XBRL · FY25",                                                                     conf: "High" },
        { name: "Revenue growth (5-yr CAGR)", value: "6.4%",                    cite: "XBRL · 5yr roll-up",                                                              conf: "High" },
        { name: "Gross margin trend",         value: "−40 bps YoY",             cite: "XBRL",                                                                            conf: "High" },
        { name: "Competitive moat",           value: "Switching cost",          cite: "§3 — \"long-term contracts and integrated workflows…\"",                          conf: "Med",  note: "Inferred from §3 language" },
      ],
      unanswered: [
        { name: "Customer concentration",     reason: "Section identified (§3.2); needs extraction" },
        { name: "Operating leverage",         reason: "Awaiting Q1 XBRL" },
      ],
    },
    capital: {
      proposed: [
        { name: "Net debt",                   value: "$1.2B",                   cite: "p.85 — balance sheet",                                                            conf: "High" },
        { name: "Net debt / EBITDA",          value: "3.1x",                    cite: "p.85 + XBRL EBITDA",                                                              conf: "High" },
        { name: "Debt maturity wall",         value: "2028 ($600M)",            cite: "p.86 — maturity schedule",                                                        conf: "High" },
        { name: "Capex intensity",            value: "5.2% of sales",           cite: "XBRL · FY25",                                                                     conf: "High" },
      ],
      blocked: [
        { name: "Pension / OPEB liability",   reason: "Pension footnote not yet filed — expected post-distribution" },
      ],
      unanswered: [
        { name: "Off-balance-sheet",          reason: "Need to cross-reference 10-K note 18" },
        { name: "Interest coverage",          reason: "Will compute on XBRL refresh" },
      ],
    },
    valuation: {
      proposed: [
        { name: "EV / EBIT",                  value: "9.4x",                    cite: "XBRL + Form 10",                                                                  conf: "High" },
        { name: "EV / EBITDA",                value: "6.8x",                    cite: "XBRL + Form 10",                                                                  conf: "High" },
        { name: "P / FCF",                    value: "11.2x",                   cite: "XBRL + Form 10",                                                                  conf: "High" },
        { name: "Dividend yield",             value: "0%",                      cite: "§5 — capital return policy",                                                      conf: "High" },
      ],
      unanswered: [
        { name: "Sum-of-parts upside",        reason: "Build from Investor Day projections" },
        { name: "Private market value",       reason: "Comparable transactions not yet pulled" },
        { name: "Mgmt guide vs. consensus",   reason: "Q1 transcript pending" },
      ],
    },
    incentives: {
      proposed: [],
      blocked: [
        { name: "Comp structure",             reason: "Spinco DEF 14A not yet filed" },
        { name: "Option vesting cliffs",      reason: "Spinco DEF 14A not yet filed" },
      ],
      unanswered: [
        { name: "CEO ownership at spin",      reason: "Use parent DEF 14A as proxy" },
        { name: "Buyback authorization",      reason: "Will activate on 8-K ingestion" },
        { name: "Insider buying (12mo)",      reason: "Form 4 history needed" },
        { name: "Parent CEO involvement",     reason: "§7 — flagged" },
        { name: "Capital allocation policy",  reason: "Investor Day partial" },
      ],
    },
  },
  /* What each bucket score WOULD be if all proposals are accepted. */
  preview: { setup: 88, business: 78, capital: 55, valuation: 62, incentives: 0 },
};

/* Second-doc demo: Investor Day deck, two CONFLICTS against existing values. */
const SAMPLE_INVESTORDAY_PROPOSALS = {
  doc: { name: "Investor Day deck", type: "PDF", pages: 47, source: "Uploaded · just now", fetched: "12s ago",
         sourceKind: "filing", trust: "primary" },
  byDim: {
    setup: {
      proposed: [],
      conflicts: [
        { name: "Forced selling estimate",
          existing: { value: "~$410M", doc: "Form 10 + peers", cite: "p.18 estimate × peer index-fund holdings" },
          proposed: { value: "~$520M", doc: "Investor Day p.31", cite: "Mgmt cited $500–540M expected index-fund outflows",  conf: "High" } },
      ],
    },
    business: {
      proposed: [
        { name: "Customer concentration", value: "Top-10 = 18% revenue", cite: "Investor Day p.12", conf: "High" },
      ],
    },
    capital: {
      proposed: [],
      conflicts: [
        { name: "Pension / OPEB liability",
          existing: { value: "$340M disclosed", doc: "Form 10 p.85", cite: "Aggregate pension obligation $340M" },
          proposed: { value: "$430M (incl. OPEB shortfall)", doc: "Investor Day p.40", cite: "CFO noted $90M OPEB funding gap on top of pension", conf: "High" } },
      ],
    },
    valuation: {
      proposed: [
        { name: "Sum-of-parts upside",       value: "+22% to current", cite: "Investor Day p.42 — peer-multiple SoTP", conf: "Med" },
        { name: "Mgmt guide vs. consensus",  value: "10% above",        cite: "Investor Day FY26 outlook",              conf: "High" },
      ],
    },
    incentives: {
      proposed: [
        { name: "Capital allocation policy", value: "Formal FCF buyback policy", cite: "Investor Day p.45 — capital return framework", conf: "High" },
      ],
    },
  },
};

/* Third-doc demo: A third-party article (Seeking Alpha-style).
   Secondary source — findings come in tagged lower-confidence, and the
   review panel surfaces a "secondary source" provenance banner. */
const SAMPLE_ARTICLE_PROPOSALS = {
  doc: {
    name: "LUMN Spinoff: Forced Selling Is Bigger Than The Street Thinks",
    type: "Article · web",
    source: "seekingalpha.com",
    fetched: "just now",
    sourceKind: "article",
    trust: "secondary",
    url: "https://seekingalpha.com/article/4683-lumn-spinoff-forced-selling",
    author: "M. Halberstam",
    publication: "Seeking Alpha · Pro Research",
    publishedOn: "May 19, 2026",
    wordCount: 2940,
  },
  byDim: {
    setup: {
      proposed: [
        { name: "Index inclusion delay", value: "~18 months post-spin",
          cite: "Author analysis citing Russell methodology and 6 peer spins (2019–2024)",
          conf: "Med", note: "Secondary — author inference from peer dataset" },
      ],
      conflicts: [
        { name: "Forced selling estimate",
          existing: { value: "~$520M", doc: "Investor Day p.31", cite: "Mgmt cited $500–540M expected index-fund outflows" },
          proposed: { value: "~$580M", doc: "Article §2 + appendix",
                      cite: "Author argues mgmt understates by ~10% based on 2024 GE Vernova / Veralto precedent",
                      conf: "Low", note: "Author opinion · not from filing" } },
      ],
    },
    business: {
      proposed: [
        { name: "Customer renewal pressure", value: "Anecdotal — flagged risk",
          cite: "Article §4 — author cites Glassdoor reviews and 2 ex-employee LinkedIn posts",
          conf: "Low", note: "Anecdotal · not corroborated by filing" },
      ],
    },
    capital: { proposed: [] },
    valuation: {
      proposed: [],
      conflicts: [
        { name: "Sum-of-parts upside",
          existing: { value: "+22% to current", doc: "Investor Day p.42", cite: "Mgmt SoTP using peer multiples" },
          proposed: { value: "+15% to current", doc: "Article §6",
                      cite: "Author SoTP using more conservative peer set (excludes 2 outliers)",
                      conf: "Med", note: "Methodology disagreement — kept-both is reasonable" } },
      ],
    },
    incentives: {
      proposed: [
        { name: "Parent CEO involvement", value: "Minimal — non-executive role",
          cite: "Article §7 — author cites recent 8-K disclosing parent CEO role limited to board observer",
          conf: "Med", note: "Verifiable against 8-K once filed" },
      ],
    },
  },
};

/* =====================================================================
   Intake — the multi-step entry surface
   identify → validate → scanning → review → ready
   ===================================================================== */

function Intake({ onCancel, onCreate }) {
  const [step, setStep] = React.useState("identify");
  const [ticker, setTicker] = React.useState("");
  const [parent, setParent] = React.useState("");
  const [type, setType] = React.useState("Spinoff");
  const [scanProgress, setScanProgress] = React.useState(0);
  const proposals = SAMPLE_FORM10_PROPOSALS;

  React.useEffect(() => {
    if (step !== "scanning") return;
    setScanProgress(0);
    let p = 0;
    const t = setInterval(() => {
      p += 6 + Math.random() * 8;
      const next = Math.min(100, p);
      setScanProgress(next);
      if (next >= 100) {
        clearInterval(t);
        setTimeout(() => setStep("review"), 500);
      }
    }, 160);
    return () => clearInterval(t);
  }, [step]);

  const stepIdx = ["identify", "validate", "scanning", "review", "ready"].indexOf(step);
  const stepNames = ["Identify", "Validate", "Initial scan", "Review", "Ready"];

  return (
    <div className="mi-bg">
      <div className={"mi-card mi-step-" + step}>
        <header className="mi-head">
          <div className="mi-head-l">
            <span className="pill blue">New analysis</span>
            <span className="mi-step-name">{stepNames[stepIdx]}</span>
          </div>
          <div className="mi-stepper">
            {stepNames.map((n, i) => (
              <span key={i} className={"dot " + (i < stepIdx ? "done" : i === stepIdx ? "active" : "")} title={n} />
            ))}
          </div>
          <button className="btn ghost mi-cancel" onClick={onCancel}>Cancel</button>
        </header>

        {step === "identify" && (
          <IdentifyStep
            ticker={ticker} setTicker={setTicker}
            parent={parent} setParent={setParent}
            type={type} setType={setType}
            onNext={() => { if (!ticker.trim()) setTicker("LUMN"); setStep("validate"); }} />
        )}
        {step === "validate" && (
          <ValidateStep
            ticker={(ticker || "LUMN").toUpperCase()}
            type={type}
            onBack={() => setStep("identify")}
            onNext={() => setStep("scanning")} />
        )}
        {step === "scanning" && (
          <ScanningStep
            ticker={(ticker || "LUMN").toUpperCase()}
            progress={scanProgress}
            doc={proposals.doc} />
        )}
        {step === "review" && (
          <ReviewStep
            ticker={(ticker || "LUMN").toUpperCase()}
            proposals={proposals}
            onBack={() => setStep("scanning")}
            onAccept={() => setStep("ready")} />
        )}
        {step === "ready" && (
          <ReadyStep
            ticker={(ticker || "LUMN").toUpperCase()}
            proposals={proposals}
            onOpenWorkspace={onCreate}
            onAddMore={() => setStep("scanning")} />
        )}
      </div>
    </div>
  );
}

/* ---- Step 1: identify ---- */
function IdentifyStep({ ticker, setTicker, parent, setParent, type, setType, onNext }) {
  return (
    <div className="mi-body">
      <h2>Open a workspace</h2>
      <div className="sub">Enter the ticker. Meridian validates it on EDGAR, fetches the most-recent Form 10 (or equivalent), and runs an initial extraction across the five buckets. Nothing gets saved until you confirm.</div>

      <div className="field">
        <label>Ticker</label>
        <input className="input mono" value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())} placeholder="LUMN" autoFocus />
      </div>
      <div className="field-grid">
        <div className="field">
          <label>Situation type</label>
          <select className="select" value={type} onChange={e => setType(e.target.value)}>
            <option>Spinoff</option>
            <option>Carve-out IPO</option>
            <option>Split-off</option>
            <option>Tracking stock</option>
            <option>Post-bankruptcy</option>
            <option>Rights offering</option>
          </select>
        </div>
        <div className="field">
          <label>Parent ticker <span className="muted" style={{ fontSize: 11 }}>auto-detected on validate</span></label>
          <input className="input mono" value={parent} onChange={e => setParent(e.target.value.toUpperCase())} placeholder="—" />
        </div>
      </div>

      <div className="mi-actions">
        <span className="muted" style={{ fontSize: 12 }}>Next: validate on EDGAR.</span>
        <button className="btn pri" onClick={onNext}>Validate on EDGAR {II.arrowR}</button>
      </div>
    </div>
  );
}

/* ---- Step 2: validate on EDGAR ---- */
function ValidateStep({ ticker, type, onBack, onNext }) {
  return (
    <div className="mi-body">
      <div className="mi-validated-pill"><span style={{ color: "var(--green)" }}>{II.check}</span> Validated on EDGAR</div>
      <h2>{ticker} · Lumen Technologies (spinco)</h2>
      <div className="sub">Parent <span className="mono">CTL</span> · CIK 0000018926 · classified as <b>{type}</b>. Here's what's available before we start extracting.</div>

      <div className="mi-val-grid">
        <ValCard label="CIK" value="0000018926" mono />
        <ValCard label="Parent" value="CTL" mono />
        <ValCard label="Form 10 on file" value="312pp · Mar 15, 2026" />
        <ValCard label="Prior 10-Ks (parent)" value="5 available" />
        <ValCard label="Recent 8-Ks" value="14 in last 12mo" />
        <ValCard label="DEF 14A · spinco" value="not yet filed" muted />
      </div>

      <div className="mi-fetch-card">
        <div className="mi-fetch-row">
          <span className="mi-fetch-doc">{II.doc} Form 10 · 312 pages</span>
          <span className="muted" style={{ fontSize: 12 }}>EDGAR · accession 0000018926-26-000041 · filed Mar 15, 2026</span>
        </div>
        <div className="mi-fetch-row">
          <span style={{ fontSize: 12.5, color: "var(--text-2)" }}>This is the primary source we'll extract from. The Q1 transcript, investor day deck, and pension footnote will be added later from the workspace.</span>
        </div>
      </div>

      <div className="mi-note">
        <span style={{ color: "var(--blue)" }}>{II.alert}</span>
        <div>
          <div className="lbl blue">What happens next</div>
          <div className="what">Meridian parses Form 10, loads XBRL facts, and proposes extractions across all 5 buckets. You review each one and accept (or reject) — nothing is written without confirmation.</div>
        </div>
      </div>

      <div className="mi-actions">
        <button className="btn ghost" onClick={onBack}>← Back</button>
        <button className="btn pri" onClick={onNext}>Run initial scan {II.arrowR}</button>
      </div>
    </div>
  );
}

function ValCard({ label, value, mono, muted }) {
  return (
    <div className="mi-val-card">
      <div className="l">{label}</div>
      <div className={"v " + (mono ? "mono " : "") + (muted ? "muted" : "")}>{value}</div>
    </div>
  );
}

/* ---- Step 3: scanning ---- */
function ScanningStep({ ticker, progress, doc }) {
  const stages = [
    { p: 12,  label: "Fetching Form 10 from EDGAR",                         meta: doc.pages + " pages · 9.4 MB" },
    { p: 28,  label: "Parsing 18 sections · table-of-contents normalized",  meta: "1,420 chunks" },
    { p: 48,  label: "Loading XBRL facts (31 financial concepts)",          meta: "FY21–FY25" },
    { p: 68,  label: "Running Haiku extractors on §1, §3, §5, §6, §7",      meta: "narrative sections" },
    { p: 86,  label: "Cross-checking against 35 Greenblatt criteria",       meta: "5 buckets" },
    { p: 100, label: "Indexing into embeddings store",                      meta: "ready for Ask-the-corpus" },
  ];
  const activeIdx = stages.findIndex(s => progress < s.p);
  const active = activeIdx === -1 ? stages.length - 1 : activeIdx;

  return (
    <div className="mi-body">
      <h2>Scanning {ticker} · Form 10</h2>
      <div className="sub">Initial extraction across the 5 buckets. Typically 60–90 seconds.</div>

      <div className="mi-progress">
        <div className="mi-progress-bar"><span style={{ width: progress + "%" }} /></div>
        <div className="mi-progress-num mono">{Math.round(progress)}<span style={{ color: "var(--text-3)" }}>/100</span></div>
      </div>

      <div className="mi-pipeline">
        {stages.map((s, i) => {
          const done = progress >= s.p;
          const cur = i === active && !done;
          return (
            <div key={i} className={"mi-pl-row " + (done ? "done " : "") + (cur ? "cur " : "")}>
              <span className="pip">
                {done && II.check}
                {cur && <span className="spin" />}
              </span>
              <div>
                <div className="lbl">{s.label}</div>
                <div className="meta">{s.meta}</div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mi-tip">
        <span className="mono" style={{ fontSize: 11, color: "var(--text-3)" }}>TIP</span>
        <span style={{ fontSize: 12.5, color: "var(--text-2)" }}>Extraction runs locally against your stored Haiku key. You can close this — Meridian will continue and notify when ready.</span>
      </div>
    </div>
  );
}

/* ---- Step 4: review extractions (the diff/preview) ---- */
function ReviewStep({ ticker, proposals, onBack, onAccept }) {
  const [acceptedCount, setAcceptedCount] = React.useState(0);
  return (
    <div className="mi-body mi-body-wide">
      <div className="mi-rev-banner">
        <span style={{ color: "var(--green)" }}>{II.check}</span>
        <div>
          <div className="lbl green">Initial scan complete</div>
          <div className="what">Found <b>extractions for 19 of 35 criteria</b>. Review what to accept — nothing is saved to {ticker}'s workspace yet.</div>
        </div>
        <div className="mi-rev-banner-r">
          <span className="muted" style={{ fontSize: 11.5 }}>Form 10 · {proposals.doc.pages}pp</span>
        </div>
      </div>

      <DocumentReviewPanel proposals={proposals} embedded onChange={setAcceptedCount} />

      <div className="mi-actions sticky">
        <button className="btn ghost" onClick={onBack}>← Re-scan</button>
        <div className="m-actions">
          <button className="btn">Reject all</button>
          <button className="btn pri" onClick={onAccept}>Apply {acceptedCount} updates {II.arrowR}</button>
        </div>
      </div>
    </div>
  );
}

/* ---- Step 5: ready — preview the workspace state + offer paths ---- */
function ReadyStep({ ticker, proposals, onOpenWorkspace, onAddMore }) {
  let activated = 0, blocked = 0, unanswered = 0;
  Object.values(proposals.byDim).forEach(b => {
    activated += (b.proposed || []).length;
    blocked += (b.blocked || []).length;
    unanswered += (b.unanswered || []).length;
  });

  return (
    <div className="mi-body">
      <div className="mi-ready-check">{II.check}</div>
      <h2 style={{ textAlign: "center" }}>{ticker} workspace ready</h2>
      <div className="sub" style={{ textAlign: "center", margin: "0 auto" }}>
        <b>{activated}</b> criteria activated · <b>{unanswered}</b> unanswered · <b>{blocked}</b> blocked awaiting docs
      </div>

      <div className="mi-ready-dims">
        {DIMS_INTAKE.map(d => {
          const pd = proposals.byDim[d.id] || {};
          const got = (pd.proposed || []).length;
          const total = got + (pd.blocked || []).length + (pd.unanswered || []).length;
          const score = proposals.preview?.[d.id] ?? 0;
          const scoreCls = score >= 80 ? "score-green" : score >= 60 ? "score-amber" : score > 0 ? "score-red" : "";
          return (
            <div key={d.id} className={"mi-ready-dim " + d.cls}>
              <span className="letter">{d.letter}</span>
              <div>
                <div className="n">{d.name}</div>
                <div className="m">{got}/{total} criteria · {got ? "draft score" : "no data yet"}</div>
              </div>
              <div className={"s " + scoreCls}>{score || "—"}</div>
            </div>
          );
        })}
      </div>

      <div className="mi-next-card">
        <div className="h">Recommended next documents</div>
        <div className="docs">
          <span className="doc-chip">{II.doc} Investor Day deck · unlocks V, B</span>
          <span className="doc-chip">{II.doc} Q1 earnings transcript · unlocks B, C</span>
          <span className="doc-chip pending">{II.clock} Pension footnote · unblocks C</span>
          <span className="doc-chip pending">{II.clock} DEF 14A · spinco · unblocks I</span>
        </div>
      </div>

      <div className="mi-actions" style={{ marginTop: 4 }}>
        <button className="btn" onClick={onAddMore}>{II.plus} Add another document now</button>
        <button className="btn pri" onClick={onOpenWorkspace}>Open workspace {II.arrowR}</button>
      </div>
    </div>
  );
}

/* =====================================================================
   DocumentReviewPanel — the diff/preview surface
   Used by intake (embedded) AND by workspace "Add document" (modal).
   ===================================================================== */
function DocumentReviewPanel({ proposals, embedded, onClose, onApply, onChange }) {
  const [decisions, setDecisions] = React.useState(() => {
    /* Default state: all proposals "accept", all conflicts "unresolved". */
    const m = {};
    Object.entries(proposals.byDim).forEach(([dim, body]) => {
      (body.proposed || []).forEach(p => { m[`${dim}::${p.name}`] = "accept"; });
      (body.conflicts || []).forEach(c => { m[`${dim}::${c.name}`] = null; });
    });
    return m;
  });
  const [activeTab, setActiveTab] = React.useState("all");

  const setDec = (k, v) => setDecisions(s => ({ ...s, [k]: s[k] === v ? null : v }));

  const totals = { proposed: 0, blocked: 0, unanswered: 0, conflicts: 0 };
  Object.values(proposals.byDim).forEach(b => {
    totals.proposed += (b.proposed || []).length;
    totals.blocked += (b.blocked || []).length;
    totals.unanswered += (b.unanswered || []).length;
    totals.conflicts += (b.conflicts || []).length;
  });
  const accepted = Object.values(decisions).filter(v => v === "accept" || v === "use-proposed" || v === "use-existing" || v === "keep-both").length;
  const rejected = Object.values(decisions).filter(v => v === "reject").length;

  React.useEffect(() => { onChange && onChange(accepted); }, [accepted]);

  const dimsToShow = activeTab === "all" ? DIMS_INTAKE : DIMS_INTAKE.filter(d => d.id === activeTab);

  return (
    <div className={"mi-rev " + (embedded ? "embedded" : "modal")}>
      {!embedded && (
        <div className="mi-rev-head">
          <div>
            <div className="t">Review · {proposals.doc.name}</div>
            <div className="m">{proposals.doc.type}{proposals.doc.pages ? ` · ${proposals.doc.pages}pp` : ""} · {proposals.doc.source}</div>
          </div>
          <button className="btn ghost" onClick={onClose}>Close ✕</button>
        </div>
      )}

      <div className="mi-rev-summary">
        <div className="mi-rev-sum-pill"><b>{totals.proposed}</b> proposed</div>
        {totals.conflicts > 0 && <div className="mi-rev-sum-pill warn"><b>{totals.conflicts}</b> conflict{totals.conflicts !== 1 ? "s" : ""}</div>}
        {totals.blocked > 0 && <div className="mi-rev-sum-pill"><b>{totals.blocked}</b> blocked</div>}
        {totals.unanswered > 0 && <div className="mi-rev-sum-pill"><b>{totals.unanswered}</b> unanswered</div>}
        <div className="mi-rev-sum-spacer" />
        <div className="mi-rev-sum-acc">{accepted} to apply · {rejected} rejected</div>
      </div>

      <div className="mi-rev-tabs">
        <button className={"mi-rev-tab " + (activeTab === "all" ? "active" : "")} onClick={() => setActiveTab("all")}>
          <span>All buckets</span><span className="ct">{totals.proposed + totals.conflicts}</span>
        </button>
        {DIMS_INTAKE.map(d => {
          const b = proposals.byDim[d.id] || {};
          const n = (b.proposed || []).length + (b.conflicts || []).length;
          return (
            <button key={d.id} className={"mi-rev-tab dim " + d.cls + (activeTab === d.id ? " active" : "")} onClick={() => setActiveTab(d.id)}>
              <span className="letter">{d.letter}</span>
              <span>{d.name}</span>
              <span className="ct">{n}</span>
            </button>
          );
        })}
      </div>

      <div className="mi-rev-list">
        {dimsToShow.map(d => {
          const body = proposals.byDim[d.id] || {};
          const propCt = (body.proposed || []).length;
          const cnfCt = (body.conflicts || []).length;
          const blkCt = (body.blocked || []).length;
          const unaCt = (body.unanswered || []).length;
          if (propCt + cnfCt + blkCt + unaCt === 0) return null;
          return (
            <div key={d.id} className={"mi-rev-dim " + d.cls}>
              <div className="mi-rev-dim-head">
                <span className="letter">{d.letter}</span>
                <span className="n">{d.name}</span>
                <span className="ct">
                  {propCt} proposed{cnfCt ? ` · ${cnfCt} conflict${cnfCt !== 1 ? "s" : ""}` : ""}{blkCt ? ` · ${blkCt} blocked` : ""}
                </span>
              </div>

              {(body.conflicts || []).map((c, i) => (
                <ConflictRow
                  key={"c"+i}
                  dimId={d.id}
                  conflict={c}
                  chosen={decisions[`${d.id}::${c.name}`]}
                  onPick={(v) => setDecisions(s => ({ ...s, [`${d.id}::${c.name}`]: v }))} />
              ))}

              {(body.proposed || []).map((p, i) => {
                const key = `${d.id}::${p.name}`;
                const state = decisions[key];
                return (
                  <div key={"p"+i} className={"mi-rev-row " + (state === "reject" ? "rejected" : state === "accept" ? "accepted" : "")}>
                    <span className={"mi-rev-pip " + (state === "reject" ? "rej" : "new")} />
                    <div className="mi-rev-crit">
                      <div className="n">{p.name}</div>
                      {p.note && <div className="m">{p.note}</div>}
                    </div>
                    <div className="mi-rev-val mono">{p.value}</div>
                    <div className="mi-rev-cite">{p.cite}</div>
                    <div className={"mi-rev-conf " + (p.conf || "").toLowerCase()}>{p.conf || "—"}</div>
                    <div className="mi-rev-controls">
                      <button className={"mi-rev-btn " + (state === "accept" ? "on" : "")} onClick={() => setDec(key, "accept")}>{II.check} Accept</button>
                      <button className={"mi-rev-btn rej " + (state === "reject" ? "on" : "")} onClick={() => setDec(key, "reject")}>Reject</button>
                    </div>
                  </div>
                );
              })}

              {(body.blocked || []).map((b, i) => (
                <div key={"b"+i} className="mi-rev-row blocked">
                  <span className="mi-rev-pip blk" />
                  <div className="mi-rev-crit">
                    <div className="n">{b.name}</div>
                    <div className="m">Blocked — {b.reason}</div>
                  </div>
                  <div className="mi-rev-val muted">—</div>
                  <div className="mi-rev-cite muted">—</div>
                  <div className="mi-rev-conf">—</div>
                  <div className="mi-rev-controls">
                    <button className="mi-rev-btn">{II.clock} Track</button>
                  </div>
                </div>
              ))}

              {(body.unanswered || []).length > 0 && (
                <div className="mi-rev-unanswered">
                  <span className="lbl">{(body.unanswered || []).length} unanswered:</span>
                  {body.unanswered.map((u, i) => (
                    <span key={i} className="mi-rev-un-chip" title={u.reason}>{u.name}</span>
                  ))}
                  <button className="mi-rev-un-fill">Fill manually →</button>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {!embedded && (
        <div className="mi-rev-foot">
          <span className="muted" style={{ fontSize: 12 }}>Selections preview the impact on bucket scores — apply to commit.</span>
          <div className="m-actions">
            <button className="btn">Reject all</button>
            <button className="btn pri" onClick={onApply}>Apply {accepted} update{accepted !== 1 ? "s" : ""} {II.arrowR}</button>
          </div>
        </div>
      )}
    </div>
  );
}

/* =====================================================================
   ConflictRow — side-by-side evidence picker
   ===================================================================== */
function ConflictRow({ conflict, dimId, chosen, onPick }) {
  return (
    <div className={"mi-rev-row conflict " + (chosen ? "resolved" : "")}>
      <span className="mi-rev-pip cnf" />
      <div className="mi-rev-crit">
        <div className="n">
          {conflict.name}
          <span className="conflict-tag">CONFLICT</span>
          {chosen && <span className="resolved-tag">resolved</span>}
        </div>
        <div className="m">Two sources disagree on this value. Pick one to apply, keep both for now, or skip and leave the existing value.</div>

        <div className="mi-conf-grid">
          <div className={"mi-conf-card existing " + (chosen === "use-existing" ? "on" : "")} onClick={() => onPick("use-existing")}>
            <div className="hd"><span>Existing</span><span className="muted">· {conflict.existing.doc}</span></div>
            <div className="vl mono">{conflict.existing.value}</div>
            <div className="ct">"{conflict.existing.cite}"</div>
          </div>
          <div className={"mi-conf-card prop " + (chosen === "use-proposed" ? "on" : "")} onClick={() => onPick("use-proposed")}>
            <div className="hd"><span>Proposed</span><span className="muted">· {conflict.proposed.doc}</span></div>
            <div className="vl mono">{conflict.proposed.value}</div>
            <div className="ct">"{conflict.proposed.cite}"</div>
          </div>
        </div>

        <div className="mi-conf-foot">
          <button className={"mi-rev-btn " + (chosen === "keep-both" ? "on" : "")} onClick={() => onPick("keep-both")}>Keep both, flag as conflict</button>
          <button className={"mi-rev-btn rej " + (chosen === "skip" ? "on" : "")} onClick={() => onPick("skip")}>Skip — leave existing</button>
        </div>
      </div>
    </div>
  );
}

/* =====================================================================
   Add-document flow — used by Workspace "Add document" button.
   source → parsing → review.

   The user can enter a document three ways:
     · Upload a file (PDF, HTML, DOCX)
     · Paste a URL (article, EDGAR filing, press release)
     · Paste raw text (transcript snippet, blog excerpt, note)

   Once a source is provided, we run a brief parsing animation, then
   route into DocumentReviewPanel — the same accept/reject/conflict
   diff surface used by the intake flow. Nothing is committed to the
   workspace until the user clicks Apply.

   Recently-suggested sources (from intake's "ready" step) live as
   quick-pick chips in the source step so the analyst can grab a
   known-pending doc with one click.
   ===================================================================== */

/* Quick-pick sources shown on the source step.
   These mirror what the intake "ready" step recommended next. */
const SOURCE_QUICKPICKS = [
  { id: "investorday", name: "Investor Day deck",      kind: "file", hint: "Saved Apr 28 · unlocks V, B",  proposals: "investorday" },
  { id: "q1call",      name: "Q1 earnings transcript", kind: "url",  hint: "From last week's call",        proposals: "investorday" },
  { id: "article1",    name: "Halberstam — 'Forced selling is bigger'", kind: "url", hint: "Seeking Alpha · May 19", proposals: "article" },
];

function AddDocumentOverlay({ onClose, onApply }) {
  const [step, setStep] = React.useState("source");      /* source | parsing | review */
  const [source, setSource] = React.useState(null);      /* { mode, ...fields } */
  const [progress, setProgress] = React.useState(0);

  /* Once a source is committed, advance to parsing and run the animation. */
  const commitSource = (s) => { setSource(s); setStep("parsing"); setProgress(0); };

  React.useEffect(() => {
    if (step !== "parsing") return;
    let p = 0;
    const t = setInterval(() => {
      p += 8 + Math.random() * 10;
      const next = Math.min(100, p);
      setProgress(next);
      if (next >= 100) {
        clearInterval(t);
        setTimeout(() => setStep("review"), 480);
      }
    }, 140);
    return () => clearInterval(t);
  }, [step]);

  /* Pick the dataset based on what was added. Article quick-pick →
     article proposals; everything else → investor-day proposals. */
  const proposals = React.useMemo(() => {
    if (!source) return SAMPLE_INVESTORDAY_PROPOSALS;
    if (source.proposalsKey === "article") return SAMPLE_ARTICLE_PROPOSALS;
    if (source.mode === "url" && /seekingalpha|substack|bloomberg|wsj|reuters/i.test(source.url || "")) {
      return SAMPLE_ARTICLE_PROPOSALS;
    }
    if (source.mode === "text" && (source.classification || "").startsWith("article")) {
      return SAMPLE_ARTICLE_PROPOSALS;
    }
    return SAMPLE_INVESTORDAY_PROPOSALS;
  }, [source]);

  const stepIdx = ["source", "parsing", "review"].indexOf(step);

  return (
    <div className="mi-bg" data-screen-label="Add document overlay">
      <div className={"mi-card mi-add-doc mi-add-step-" + step}>
        <header className="mi-head">
          <div className="mi-head-l">
            <span className="pill blue">Add document</span>
            <span className="mi-step-name">
              {step === "source" ? "Choose source" :
               step === "parsing" ? "Parsing & comparing" : "Review findings"}
            </span>
          </div>
          <div className="mi-stepper">
            {["Source", "Parse", "Review"].map((n, i) => (
              <span key={i} className={"dot " + (i < stepIdx ? "done" : i === stepIdx ? "active" : "")} title={n} />
            ))}
          </div>
          <button className="btn ghost mi-cancel" onClick={onClose}>Cancel</button>
        </header>

        {step === "source" && (
          <DocumentSourceStep
            onCommit={commitSource}
            onPickQuick={(qp) => commitSource({
              mode: qp.kind, name: qp.name, hint: qp.hint,
              quickpick: true, proposalsKey: qp.proposals,
            })} />
        )}
        {step === "parsing" && (
          <DocumentParsingStep source={source} progress={progress} />
        )}
        {step === "review" && (
          <DocumentReviewWithProvenance
            proposals={proposals}
            source={source}
            onBack={() => setStep("source")}
            onApply={() => { onApply && onApply(); onClose(); }} />
        )}
      </div>
    </div>
  );
}

/* ---- Source step: three input modes + quick picks ---- */
function DocumentSourceStep({ onCommit, onPickQuick }) {
  const [mode, setMode] = React.useState("file");
  const [fileName, setFileName] = React.useState("");
  const [fileMeta, setFileMeta] = React.useState("");
  const [url, setUrl] = React.useState("");
  const [urlPreview, setUrlPreview] = React.useState(null);
  const [text, setText] = React.useState("");
  const [textKind, setTextKind] = React.useState("article-excerpt");
  const [docKind, setDocKind] = React.useState("auto");
  const [dragOver, setDragOver] = React.useState(false);

  /* Faux "fetch metadata" when a recognisable URL is pasted. */
  React.useEffect(() => {
    if (mode !== "url" || !url.trim()) { setUrlPreview(null); return; }
    const u = url.trim();
    if (!/^https?:\/\//.test(u)) { setUrlPreview(null); return; }
    const host = (u.match(/^https?:\/\/(?:www\.)?([^/]+)/) || [, "source"])[1];
    const guesses = {
      "seekingalpha.com":   { title: "LUMN Spinoff: Forced Selling Is Bigger Than The Street Thinks", pub: "Seeking Alpha", author: "M. Halberstam", date: "May 19, 2026", kind: "Article", trust: "secondary" },
      "sec.gov":            { title: "LUMN · 8-K · Material agreement amendment", pub: "EDGAR", author: "Lumen Inc.", date: "May 18, 2026", kind: "Filing", trust: "primary" },
      "substack.com":       { title: "Independent research note · LUMN deep dive", pub: "Substack", author: "Independent", date: "May 12, 2026", kind: "Article", trust: "secondary" },
    };
    const m = guesses[host] || { title: u.length > 70 ? u.slice(0, 67) + "…" : u, pub: host, author: "—", date: "—", kind: "Article", trust: "secondary" };
    setUrlPreview({ host, ...m });
  }, [url, mode]);

  /* File upload: real file selected via input — but we don't parse, just
     capture name + bytes for the demo. */
  const onFile = (f) => {
    if (!f) return;
    setFileName(f.name);
    setFileMeta((f.type.split("/")[1] || "file").toUpperCase() + " · " + Math.max(1, Math.round(f.size / 1024)) + " KB");
  };

  const canSubmit =
    (mode === "file" && fileName) ||
    (mode === "url" && url.trim() && urlPreview) ||
    (mode === "text" && text.trim().length > 80);

  const submit = () => {
    if (!canSubmit) return;
    if (mode === "file") onCommit({ mode: "file", name: fileName, meta: fileMeta, docKind });
    else if (mode === "url") onCommit({ mode: "url", url, ...urlPreview });
    else onCommit({ mode: "text", text, classification: textKind, length: text.length });
  };

  return (
    <div className="mi-body mi-add-source">
      <h2>Add a document</h2>
      <div className="sub">
        Upload a file, paste a URL, or paste an excerpt. Meridian parses it, compares each
        extraction against existing criteria, and surfaces conflicts side-by-side — you accept,
        reject, or keep both. Nothing is written until you confirm.
      </div>

      <div className="mi-trust-note">
        <span className="dot" />
        <div>
          <b>Trust model:</b> primary filings (10-K, 10-Q, 8-K, DEF 14A) can promote to a criterion
          on Accept. Articles, notes, and pasted excerpts are tagged <span className="mono">secondary</span> —
          they surface as candidates and never silently overwrite a primary value.
        </div>
      </div>

      <div className="mi-mode-tabs">
        {[
          { k: "file", l: "Upload PDF / HTML",  s: "10-K, transcript, deck, saved article" },
          { k: "url",  l: "Paste URL",          s: "Article link, EDGAR filing, press release" },
          { k: "text", l: "Paste text",         s: "Transcript snippet, excerpt, note" },
        ].map(t => (
          <button key={t.k} className={"mi-mode-tab " + (mode === t.k ? "active" : "")} onClick={() => setMode(t.k)}>
            <div className="l">{t.l}</div>
            <div className="s">{t.s}</div>
          </button>
        ))}
      </div>

      {mode === "file" && (
        <div
          className={"mi-drop " + (dragOver ? "over " : "") + (fileName ? "filled" : "")}
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={e => { e.preventDefault(); setDragOver(false); onFile(e.dataTransfer.files[0]); }}
        >
          {!fileName ? (
            <>
              <div className="mi-drop-ico">{II.doc}</div>
              <div className="mi-drop-t">Drop a file here, or <label className="mi-drop-link">browse<input type="file" accept=".pdf,.html,.htm,.docx,.txt" onChange={e => onFile(e.target.files[0])} hidden /></label></div>
              <div className="mi-drop-s">PDF · HTML · DOCX · TXT  ·  up to 30 MB</div>
            </>
          ) : (
            <>
              <div className="mi-drop-file">
                <span className="ico">{II.doc}</span>
                <div>
                  <div className="n">{fileName}</div>
                  <div className="m">{fileMeta}</div>
                </div>
                <button className="btn ghost" style={{ padding: "5px 10px", fontSize: 12 }} onClick={() => { setFileName(""); setFileMeta(""); }}>Remove</button>
              </div>
              <div className="mi-drop-kind">
                <span className="lbl">Classify as</span>
                {[
                  { k: "auto",      l: "Auto-detect"   },
                  { k: "filing",    l: "Primary filing" },
                  { k: "transcript",l: "Transcript"    },
                  { k: "deck",      l: "Investor deck" },
                  { k: "article",   l: "Article / note" },
                ].map(o => (
                  <button key={o.k} className={"mi-kind-chip " + (docKind === o.k ? "on" : "")} onClick={() => setDocKind(o.k)}>{o.l}</button>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {mode === "url" && (
        <div className="mi-url-wrap">
          <input
            className="input mono"
            placeholder="https://seekingalpha.com/article/… or https://www.sec.gov/…"
            value={url}
            onChange={e => setUrl(e.target.value)}
            autoFocus
          />
          {urlPreview && (
            <div className="mi-url-preview">
              <div className="mi-url-fav">
                <span>{II.link || II.doc}</span>
              </div>
              <div className="mi-url-body">
                <div className="mi-url-title">{urlPreview.title}</div>
                <div className="mi-url-meta">
                  <span>{urlPreview.pub}</span>
                  <span className="dot-sep">·</span>
                  <span>{urlPreview.author}</span>
                  <span className="dot-sep">·</span>
                  <span>{urlPreview.date}</span>
                </div>
              </div>
              <span className={"mi-trust-pill " + (urlPreview.trust === "primary" ? "primary" : "secondary")}>
                {urlPreview.trust === "primary" ? "Primary" : "Secondary"}
              </span>
            </div>
          )}
          {url && !urlPreview && <div className="mi-url-hint muted">Enter a full URL starting with <span className="mono">https://</span> to fetch metadata.</div>}
        </div>
      )}

      {mode === "text" && (
        <div className="mi-text-wrap">
          <textarea
            className="input"
            rows={9}
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Paste the article body, transcript section, or note here.&#10;&#10;Meridian classifies it, runs the 5-bucket extractors against the text, and shows you what it proposes."
            style={{ fontFamily: "inherit", resize: "vertical", lineHeight: 1.55 }}
          />
          <div className="mi-text-foot">
            <div className="mi-text-meta mono">
              <span>{text.length.toLocaleString()} chars</span>
              <span>·</span>
              <span>{text.trim() ? text.trim().split(/\s+/).length.toLocaleString() : 0} words</span>
              {text.length > 80 && <span className="ok">· enough to extract</span>}
              {text.length > 0 && text.length <= 80 && <span className="warn">· too short — paste at least a paragraph</span>}
            </div>
            <div className="mi-text-kind">
              <span className="lbl">This is a</span>
              {[
                { k: "article-excerpt", l: "Article excerpt" },
                { k: "transcript",      l: "Transcript chunk" },
                { k: "note",            l: "My note / memo" },
                { k: "press",           l: "Press release" },
              ].map(o => (
                <button key={o.k} className={"mi-kind-chip " + (textKind === o.k ? "on" : "")} onClick={() => setTextKind(o.k)}>{o.l}</button>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="mi-quickpicks">
        <div className="mi-quickpicks-h">Pending in workspace · one-click add</div>
        <div className="mi-quickpicks-row">
          {SOURCE_QUICKPICKS.map(qp => (
            <button key={qp.id} className="mi-quickpick" onClick={() => onPickQuick(qp)}>
              <span className="ico">{qp.kind === "file" ? II.doc : II.link || II.doc}</span>
              <div>
                <div className="n">{qp.name}</div>
                <div className="m">{qp.hint}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="mi-actions sticky">
        <span className="muted" style={{ fontSize: 12 }}>
          {canSubmit ? "Next: parse the document and compare to existing criteria." : "Provide a source to continue."}
        </span>
        <button className={"btn " + (canSubmit ? "pri" : "")} disabled={!canSubmit} onClick={submit}>
          Parse and compare {II.arrowR}
        </button>
      </div>
    </div>
  );
}

/* ---- Parsing step: adaptive pipeline by source kind ---- */
function DocumentParsingStep({ source, progress }) {
  /* Pipeline stages adapt to source mode. */
  const pipelines = {
    file: [
      { p: 14,  label: "Extracting text from " + (source?.name || "file"), meta: "PDF / HTML decoder" },
      { p: 32,  label: "Structuring sections",                              meta: "tables of contents, footnotes, exhibits" },
      { p: 54,  label: "Classifying document kind",                         meta: source?.docKind && source.docKind !== "auto" ? "user-tagged: " + source.docKind : "auto-classifier" },
      { p: 74,  label: "Running 5-bucket extractors",                       meta: "S · B · C · V · I" },
      { p: 92,  label: "Comparing to existing criteria",                    meta: "Diffing against workspace values" },
      { p: 100, label: "Indexing for Ask-the-corpus",                       meta: "embeddings + citations" },
    ],
    url: [
      { p: 14,  label: "Fetching page",                                     meta: source?.host || (source?.url || "").slice(0, 48) },
      { p: 30,  label: "Stripping chrome (ads, nav, comments)",             meta: "Readability extractor" },
      { p: 50,  label: "Classifying source authority",                      meta: source?.trust === "primary" ? "Primary filing" : "Secondary · article" },
      { p: 72,  label: "Running 5-bucket extractors",                       meta: "S · B · C · V · I" },
      { p: 92,  label: "Comparing to existing criteria",                    meta: "Diffing against workspace values" },
      { p: 100, label: "Indexed",                                           meta: "ready for review" },
    ],
    text: [
      { p: 18,  label: "Tokenizing pasted text",                            meta: (source?.length || 0).toLocaleString() + " chars" },
      { p: 38,  label: "Classifying as " + (source?.classification || "excerpt"), meta: "context-aware classifier" },
      { p: 60,  label: "Running 5-bucket extractors",                       meta: "S · B · C · V · I" },
      { p: 84,  label: "Comparing to existing criteria",                    meta: "Diffing against workspace values" },
      { p: 100, label: "Ready for review",                                  meta: "—" },
    ],
  };
  const stages = pipelines[source?.mode] || pipelines.file;
  const activeIdx = stages.findIndex(s => progress < s.p);
  const active = activeIdx === -1 ? stages.length - 1 : activeIdx;

  const headline =
    source?.mode === "url"  ? source.title || source.url :
    source?.mode === "text" ? "Pasted text · " + (source.classification || "excerpt") :
                              source?.name || "Document";

  return (
    <div className="mi-body">
      <h2>Parsing & comparing</h2>
      <div className="sub">{headline}</div>

      <div className="mi-progress">
        <div className="mi-progress-bar"><span style={{ width: progress + "%" }} /></div>
        <div className="mi-progress-num mono">{Math.round(progress)}<span style={{ color: "var(--text-3)" }}>/100</span></div>
      </div>

      <div className="mi-pipeline">
        {stages.map((s, i) => {
          const done = progress >= s.p;
          const cur = i === active && !done;
          return (
            <div key={i} className={"mi-pl-row " + (done ? "done " : "") + (cur ? "cur " : "")}>
              <span className="pip">
                {done && II.check}
                {cur && <span className="spin" />}
              </span>
              <div>
                <div className="lbl">{s.label}</div>
                <div className="meta">{s.meta}</div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mi-tip">
        <span className="mono" style={{ fontSize: 11, color: "var(--text-3)" }}>NOTE</span>
        <span style={{ fontSize: 12.5, color: "var(--text-2)" }}>
          {source?.trust === "secondary" || source?.mode === "text"
            ? "Findings from this source will be tagged secondary. They won't auto-replace primary-filing values — they surface as candidates next to existing ones."
            : "Findings from this filing can promote to criterion values on Accept."}
        </span>
      </div>
    </div>
  );
}

/* ---- Review step wrapper: source provenance + DocumentReviewPanel ---- */
function DocumentReviewWithProvenance({ proposals, source, onBack, onApply }) {
  const isSecondary = source?.trust === "secondary" || proposals.doc?.trust === "secondary";

  return (
    <>
      <div style={{ padding: "12px 22px 0" }}>
        <div className={"mi-provenance " + (isSecondary ? "secondary" : "primary")}>
          <span className="mi-prov-ico">{II.doc}</span>
          <div className="mi-prov-body">
            <div className="mi-prov-title">{proposals.doc.name}</div>
            <div className="mi-prov-meta">
              <span>{proposals.doc.type}</span>
              {proposals.doc.pages && <><span className="sep">·</span><span>{proposals.doc.pages}pp</span></>}
              <span className="sep">·</span>
              <span>{proposals.doc.source}</span>
              {proposals.doc.author && <><span className="sep">·</span><span>{proposals.doc.author}</span></>}
              {proposals.doc.publishedOn && <><span className="sep">·</span><span>{proposals.doc.publishedOn}</span></>}
            </div>
          </div>
          <div className="mi-prov-tags">
            <span className={"mi-trust-pill " + (isSecondary ? "secondary" : "primary")}>{isSecondary ? "Secondary source" : "Primary filing"}</span>
            {proposals.doc.url && <a className="mi-prov-link" href={proposals.doc.url} target="_blank" rel="noreferrer">Open source ↗</a>}
          </div>
        </div>
        {isSecondary && (
          <div className="mi-secondary-note">
            Findings below are <b>candidate values</b>, not authoritative. Accept promotes a value only when the existing source is empty; on a conflict you'll be asked to choose explicitly.
          </div>
        )}
      </div>

      <DocumentReviewPanel proposals={proposals} embedded onChange={() => {}} />

      <div className="mi-actions sticky" style={{ padding: "16px 22px 18px", margin: 0, borderTop: "1px solid rgba(255,255,255,0.06)" }}>
        <button className="btn ghost" onClick={onBack}>← Change source</button>
        <div className="m-actions">
          <button className="btn">Reject all</button>
          <button className="btn pri" onClick={onApply}>Apply to workspace {II.arrowR}</button>
        </div>
      </div>
    </>
  );
}

window.MeridianIntake = {
  Intake, DocumentReviewPanel, ConflictRow, AddDocumentOverlay,
  DocumentSourceStep, DocumentParsingStep, DocumentReviewWithProvenance,
  SAMPLE_FORM10_PROPOSALS, SAMPLE_INVESTORDAY_PROPOSALS, SAMPLE_ARTICLE_PROPOSALS,
};

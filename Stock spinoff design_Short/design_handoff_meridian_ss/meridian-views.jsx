/* global React, SpinoffUI, MeridianData, MeridianWorkspaceParts */
const { I: Ic } = SpinoffUI;
const { DIMS: DIMS2, COMPOSITE: COMP } = MeridianData;
const { DimCard, DimDetail, WorkspaceHeader, CommitteePanel, DocsPane, ActivityFeed, QuarterlyReviewPanel } = MeridianWorkspaceParts;

/* ====================================================================
   The Workspace view — the heart of Meridian-SS
==================================================================== */
function Workspace({ goNewIntake, openQuarterly }) {
  const [focused, setFocused] = React.useState(null);
  const [committee, setCommittee] = React.useState(false);
  const [coachOpen, setCoachOpen] = React.useState(true);
  const [quarterly, setQuarterly] = React.useState(!!openQuarterly);
  const [quarterlyDismissed, setQuarterlyDismissed] = React.useState(false);

  React.useEffect(() => { if (openQuarterly) setQuarterly(true); }, [openQuarterly]);

  return (
    <div>
      <WorkspaceHeader />

      {!quarterlyDismissed && (
        <div className="m-quarterly-banner" onClick={() => setQuarterly(v => !v)}>
          <span style={{ color: "var(--d-business)" }}>{Ic.doc}</span>
          <div>
            <div className="lbl">Quarterly review available</div>
            <div className="what">LUMN Q1 2026 10-Q filed May 9 · auto-ingested · <b>6 criteria moved</b>, 3 promises reconciled.</div>
            <div className="why">Composite drifted 65 → 56 before committee re-run. Pension footnote (was blocked) is now extracted.</div>
          </div>
          <div className="m-actions">
            <span className="m-watch-pill"><span className="dot" />Watching EDGAR</span>
            <button className="btn pri">{quarterly ? "Close" : "Review changes"} {Ic.arrowR}</button>
          </div>
        </div>
      )}

      {quarterly && (
        <QuarterlyReviewPanel
          onClose={() => { setQuarterly(false); setQuarterlyDismissed(true); }}
          onRunCommittee={() => setCommittee(true)} />
      )}

      {coachOpen && (
        <div className="m-tc" style={{ display: "grid", gridTemplateColumns: "auto 1fr auto", gap: 14, alignItems: "center" }}>
          <span style={{ color: "var(--amber)" }}>{Ic.alert}</span>
          <div>
            <div className="lbl">Tendency coach</div>
            <div className="what">You've re-run the committee 3× this week without adding a new document. Pattern detected: <b>#AnalysisParalysis</b>.</div>
            <div className="muted" style={{ fontSize: 12.5, marginTop: 4 }}>Suggested move: chase the pension footnote — it unblocks the only criterion blocking your Capital score.</div>
          </div>
          <button className="btn ghost" style={{ padding: "6px 10px" }} onClick={() => setCoachOpen(false)}>Dismiss</button>
        </div>
      )}

      <div className="m-guide">
        <span style={{ color: "var(--blue)" }}>{Ic.arrowR}</span>
        <div>
          <div className="lbl blue">Next best action</div>
          <div className="what">Capital is your weakest dimension. The pension footnote (expected today) unblocks 1 criterion and stabilizes the score.</div>
          <div className="why">Likely composite move: +6 to +12. Two other criteria in Incentives remain blocked until DEF 14A files post-distribution.</div>
        </div>
        <div className="m-actions">
          <button className="btn">Check EDGAR {Ic.arrowUR}</button>
          <button className="btn pri">Upload footnote</button>
        </div>
      </div>

      {/* Primary workspace actions */}
      <div className="m-actions" style={{ marginBottom: 16, justifyContent: "flex-end" }}>
        <button className="btn">{Ic.plus} Add document</button>
        <button className="btn">{Ic.chat} Ask the corpus</button>
        <button className="btn">{Ic.copy2} Memo</button>
        <button className="btn pri" onClick={() => setCommittee(true)}>Run committee {Ic.arrowR}</button>
      </div>

      <div className="m-dim-grid">
        {DIMS2.map(d => (
          <DimCard key={d.id} dim={d} focused={focused === d.id} onFocus={(id) => setFocused(focused === id ? null : id)} />
        ))}
      </div>

      {focused && <DimDetail dimId={focused} onClose={() => setFocused(null)} />}

      {committee && <CommitteePanel onClose={() => setCommittee(false)} />}

      <div className="m-rail">
        <ActivityFeed />
        <div>
          <DocsPane />
          <DecisionJournal />
        </div>
      </div>
    </div>
  );
}

/* ====================================================================
   Decision journal — inline panel (called from workspace right rail)
==================================================================== */
function DecisionJournal() {
  const [conv, setConv] = React.useState(6);
  return (
    <div className="m-journal" style={{ marginTop: 18 }}>
      <div className="h">Decision journal</div>
      <div className="field">
        <label>Primary driver</label>
        <input className="input" defaultValue="Setup played out · forced selling well-quantified" />
      </div>
      <div className="field">
        <label>Conviction</label>
        <div className="slider-row">
          <div className="ticks">
            {[1,2,3,4,5,6,7,8,9,10].map(n => (
              <span key={n} className={"tick " + (n === conv ? "active" : "")} onClick={() => setConv(n)}>{n}</span>
            ))}
          </div>
          <span className="muted" style={{ fontSize: 12 }}>{conv}/10</span>
        </div>
      </div>
      <div className="field">
        <label>Biggest acknowledged risk</label>
        <input className="input" defaultValue="Pension funding shortfall not yet quantified" />
      </div>
      <div className="field">
        <label>Core thesis (1–3 sentences)</label>
        <textarea className="input" rows={3} style={{ resize: "vertical", fontFamily: "inherit" }}
          defaultValue="Forced-selling spinoff with above-average business quality at attractive multiples. Pension uncertainty caps initial sizing at 5%. Re-evaluate at footnote." />
      </div>
      <div className="m-actions" style={{ justifyContent: "flex-end" }}>
        <button className="btn">Reject</button>
        <button className="btn">Watch</button>
        <button className="btn pri">Log invest</button>
      </div>
    </div>
  );
}

/* ====================================================================
   Portfolio view — list of workspaces
==================================================================== */
function Portfolio({ onOpen, onNew }) {
  const positions = [
    { ticker: "LUMN",  name: "Lumen Spinco",  composite: 65, state: "researching", states: { S: 88, B: 82, C: 45, V: 60, I: 48 }, played: { S: true }, blocked: 4, last: "Today" },
    { ticker: "VNTR",  name: "Vontier",       composite: 58, state: "invested", states: { S: "Played", B: 62, C: 45, V: 60, I: 48 }, played: { S: true }, blocked: 1, last: "May 9" },
    { ticker: "STHO",  name: "Stericycle Hldg",composite: 82, state: "invested", states: { S: "Played", B: 86, C: 78, V: 82, I: 80 }, played: { S: true }, blocked: 0, last: "Q4 '25" },
    { ticker: "LTRPA", name: "Liberty Tripadvisor", composite: 74, state: "memo-ready", states: { S: 80, B: 78, C: 62, V: 70, I: 80 }, played: {}, blocked: 0, last: "May 18" },
  ];

  const scoreCls = (s) => typeof s === "number" ? (s >= 80 ? "score-green" : s >= 60 ? "score-amber" : "score-red") : "";

  return (
    <div>
      <div className="m-page-head">
        <div>
          <h1 className="h1">Portfolio</h1>
          <div className="sub">7 invested · 14 on watchlist · 3 pending review</div>
        </div>
        <div className="m-actions">
          <button className="btn">{Ic.search} Find ticker</button>
          <button className="btn pri" onClick={onNew}>{Ic.plus} New analysis</button>
        </div>
      </div>

      <div className="kpi-grid" style={{ marginBottom: 22 }}>
        <div className="kpi"><div className="label">Invested</div><div className="val">7</div></div>
        <div className="kpi"><div className="label">Watchlist</div><div className="val">14</div></div>
        <div className="kpi"><div className="label">Pending review</div><div className="val amber">3</div></div>
        <div className="kpi"><div className="label">Avg score</div><div className="val">71</div></div>
      </div>

      <div className="m-portfolio">
        {positions.map(p => (
          <div className="m-port-card" key={p.ticker} onClick={() => onOpen(p.ticker)}>
            <div className="top">
              <span className="ticker">{p.ticker}</span>
              <span className="name">{p.name}</span>
              <span className={"composite " + scoreCls(p.composite)}>{p.composite}</span>
            </div>
            <div className="dims-strip">
              {DIMS2.map(d => {
                const v = p.states[d.letter];
                const isText = typeof v !== "number";
                return (
                  <div key={d.id} className="seg" style={{ background: `var(--d-${d.id}-soft)` }}>
                    <span className="letter" style={{ color: `var(--d-${d.id})`, fontWeight: 600 }}>{d.letter}</span>
                    <span className={"v " + (isText ? "" : scoreCls(v))}>{isText ? v : v}</span>
                  </div>
                );
              })}
            </div>
            <div className="meta-row">
              <span className={"m-state " + p.state}><span className="dot" />{
                p.state === "researching" ? "Researching" :
                p.state === "memo-ready"  ? "Memo ready" : "Invested"
              }</span>
              <span>{p.blocked ? `${p.blocked} blocked · ` : ""}Last activity {p.last}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ====================================================================
   Updates inbox — workspace-state deltas
==================================================================== */
function Updates({ onOpen }) {
  const updates = [
    { ticker: "VNTR", type: "quarterly", title: "VNTR Q1 2026 · 10-Q filed",
      meta: "May 9 · 5-dim deltas detected",
      changes: [
        { dim: "business",  delta: -8 },
        { dim: "capital",   delta: -15 },
        { dim: "valuation", delta: -5 },
        { dim: "incentives",delta: -12 },
      ],
      ico: "doc", action: "review", body: "Composite dropped 60 → 48. 2 missed promises flagged." },
    { ticker: "LUMN", type: "8-K", title: "LUMN · 8-K filed May 18 · material agreement amendment",
      meta: "Add the 8-K and any related news, then re-run committee",
      ico: "bell", action: "ingest", body: "Capital dimension may be affected." },
    { ticker: "LUMN", type: "pending-doc", title: "LUMN · pension footnote expected today",
      meta: "Flagged Apr 22 as missing — has it been disclosed?",
      ico: "clock", action: "chase", body: "Blocking 1 Capital criterion in the workspace." },
    { ticker: "LTRPA", type: "promise", title: "LTRPA · mgmt SoTP commitment due",
      meta: "Promised at Q4 investor day; quarterly review approaching",
      ico: "bookmark", action: "track", body: "Promise tracker due in 11 days." },
  ];

  const dimByLetter = Object.fromEntries(DIMS2.map(d => [d.letter, d]));
  const dimById     = Object.fromEntries(DIMS2.map(d => [d.id, d]));
  const ico = { doc: Ic.doc, bell: Ic.bell, clock: Ic.clock, bookmark: Ic.bookmark };

  return (
    <div>
      <div className="m-page-head">
        <div>
          <h1 className="h1">Updates <span className="muted" style={{ fontWeight: 400, fontSize: 14 }}>· workspace deltas</span></h1>
          <div className="sub">Quarterly filings, 8-Ks, missing-doc reminders, and promise deadlines that change something in a company's workspace.</div>
        </div>
        <div className="m-actions">
          <button className="btn">Filter</button>
          <button className="btn">Mark all read</button>
        </div>
      </div>

      <div className="upd-tabs">
        {[{k:"All",n:4},{k:"Quarterly",n:1},{k:"8-K",n:1},{k:"Pending docs",n:1},{k:"Promises",n:1}].map(t => (
          <button key={t.k} className={"upd-tab " + (t.k === "All" ? "active" : "")}>{t.k} · {t.n}</button>
        ))}
      </div>

      {updates.map((u, i) => (
        <div key={i} className="upd-card" onClick={() => onOpen(u.ticker, u.type === "quarterly" ? { quarterly: true } : {})} style={{ cursor: "pointer" }}>
          <div className={"head " + (u.ico === "bell" ? "bell" : u.ico === "clock" ? "clock" : "")}>
            <div className="ico-wrap">{ico[u.ico]}</div>
            <div>
              <div className="title">{u.title}</div>
              <div className="sub">{u.meta}</div>
            </div>
            {u.action === "review" && <span className="pill amber">Review</span>}
            {u.action === "ingest" && <span className="pill blue">Ingest</span>}
            {u.action === "chase"  && <span className="pill amber">Chase</span>}
            {u.action === "track"  && <span className="pill green">Track</span>}
          </div>
          {u.changes && (
            <div style={{ padding: "0 20px 18px" }}>
              <div className="muted" style={{ fontSize: 12.5, marginBottom: 8 }}>Dimension impact</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8 }}>
                {DIMS2.map(d => {
                  const c = u.changes.find(x => x.dim === d.id);
                  return (
                    <div key={d.id} className={"score-tile " + d.cls} style={{ padding: "10px 12px" }}>
                      <div className="label">{d.name.split(" ")[0]}</div>
                      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
                        <div className="val" style={{ fontSize: 18 }}>{c ? "·" : "—"}</div>
                        {c && <span className="delta red" style={{ fontSize: 14 }}>{c.delta}</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="muted" style={{ fontSize: 13, marginTop: 10 }}>{u.body}</div>
            </div>
          )}
          {!u.changes && (
            <div className="actions">
              {u.action === "ingest" && (<>
                <button className="btn">Dismiss</button>
                <button className="btn pri">Add documents and re-run {Ic.arrowUR}</button>
              </>)}
              {u.action === "chase" && (<>
                <button className="btn">Still missing</button>
                <button className="btn">Check EDGAR {Ic.arrowUR}</button>
                <button className="btn pri">Upload now {Ic.arrowUR}</button>
              </>)}
              {u.action === "track" && (<>
                <button className="btn">Snooze 30 days</button>
                <button className="btn pri">Mark fulfilled</button>
              </>)}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/* ====================================================================
   New Analysis intake — 30-second modal-ish single screen
==================================================================== */
function Intake({ onCancel, onCreate }) {
  const [ticker, setTicker] = React.useState("");
  const [parent, setParent] = React.useState("");
  const [type, setType] = React.useState("Spinoff");
  const [validated, setValidated] = React.useState(false);

  return (
    <div className="m-intake-bg">
      <div className="m-intake">
        <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
          <span className="pill blue">New analysis</span>
          <button className="btn ghost" style={{ padding: "4px 8px", fontSize: 12 }} onClick={onCancel}>Cancel</button>
        </div>
        <h2>Open a workspace</h2>
        <div className="sub">A 30-second intake. Once created, the workspace will guide you through documents, evidence, committee, and memo.</div>

        <div className="field">
          <label>Ticker</label>
          <input className="input" value={ticker} onChange={e => { setTicker(e.target.value); setValidated(false); }} placeholder="e.g. LUMN" />
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
            <label>Parent ticker <span className="muted" style={{ fontSize: 11 }}>auto-detected</span></label>
            <input className="input" value={parent} onChange={e => setParent(e.target.value)} placeholder="—" />
          </div>
        </div>
        <div className="field">
          <label>Seed thesis URL <span className="muted" style={{ fontSize: 11 }}>optional · newsletter or writeup</span></label>
          <input className="input" placeholder="https://…" />
        </div>

        {validated && (
          <div className="m-guide good" style={{ marginTop: 4, marginBottom: 14 }}>
            <span style={{ color: "var(--green)" }}>{Ic.check}</span>
            <div>
              <div className="lbl green">Validated on EDGAR</div>
              <div className="what">{ticker.toUpperCase()} · {parent ? `parent ${parent.toUpperCase()} · ` : ""}CIK detected · 5 prior filings available for bulk ingest</div>
            </div>
            <div />
          </div>
        )}

        <div className="m-actions" style={{ justifyContent: "space-between", marginTop: 10 }}>
          <span className="muted" style={{ fontSize: 12 }}>You can add documents from the workspace.</span>
          {!validated
            ? <button className="btn pri" onClick={() => setValidated(true)}>Validate on EDGAR {Ic.arrowR}</button>
            : <button className="btn pri" onClick={onCreate}>Open workspace {Ic.arrowR}</button>}
        </div>
      </div>
    </div>
  );
}

/* ====================================================================
   Dev / settings — DB inspector + API keys
==================================================================== */
function Dev() {
  const rows = [
    { ticker: "LUMN",  docs: 5, notes: 12, criteria: "26 / 35", last: "Today" },
    { ticker: "VNTR",  docs: 7, notes: 8,  criteria: "33 / 35", last: "May 9" },
    { ticker: "STHO",  docs: 9, notes: 14, criteria: "35 / 35", last: "Q4 '25" },
    { ticker: "LTRPA", docs: 6, notes: 4,  criteria: "31 / 35", last: "May 18" },
  ];
  return (
    <div>
      <div className="m-page-head">
        <div>
          <h1 className="h1">Dev <span className="muted" style={{ fontWeight: 400, fontSize: 14 }}>· local store</span></h1>
          <div className="sub">meridian.db (SQLite) — documents, notes, decision logs, XBRL & Haiku sidecars.</div>
        </div>
      </div>

      <h3 className="section-h">API keys</h3>
      <div className="card card-pad" style={{ marginBottom: 22 }}>
        <div className="field-grid">
          <div className="field">
            <label>OpenAI · embeddings (ada-002)</label>
            <input className="input" defaultValue="sk-…hidden" type="password" />
          </div>
          <div className="field">
            <label>Anthropic · Sonnet / Opus / Haiku</label>
            <input className="input" defaultValue="sk-ant-…hidden" type="password" />
          </div>
        </div>
        <div className="muted" style={{ fontSize: 12, marginTop: 10 }}>Keys are kept in <span className="mono">os.environ</span> for the session only — not persisted.</div>
      </div>

      <h3 className="section-h">Tickers in store</h3>
      <div className="card">
        <table className="m-dev-table">
          <thead><tr><th>Ticker</th><th>Documents</th><th>Notes</th><th>Coverage</th><th>Last activity</th><th></th></tr></thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.ticker}>
                <td><span className="mono" style={{ fontWeight: 600 }}>{r.ticker}</span></td>
                <td className="muted">{r.docs}</td>
                <td className="muted">{r.notes}</td>
                <td className="mono">{r.criteria}</td>
                <td className="muted">{r.last}</td>
                <td className="actions">
                  <button className="btn ghost" style={{ padding: "5px 10px", fontSize: 12, color: "var(--red)" }}>Delete {r.ticker}</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

window.MeridianViews = { Workspace, Portfolio, Updates, Intake, Dev };

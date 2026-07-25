/* global React, SpinoffUI */
const { I } = SpinoffUI;

/* ===========================================================
   ALT A · "Dimension Workspace"
   5 dimensions are the workspace. Each card shows checklist
   sub-questions, "driven by" docs, and the next action.
=========================================================== */
function AltA() {
  const [focus, setFocus] = React.useState("Capital");
  const dims = [
    {
      name: "Setup", letter: "S", score: "Played", scoreText: true, color: "g",
      qs: [
        { s: "done", t: "Index exclusion confirmed (3 indices)" },
        { s: "done", t: "Insider forced sale window <12mo" },
        { s: "done", t: "Cap below institutional minimum" },
      ],
      docs: ["Form 10 p.12", "CTL 10-K p.4"],
      next: "Complete — all setup questions answered.",
      nextHas: false,
    },
    {
      name: "Business", letter: "B", score: 82, color: "g",
      qs: [
        { s: "done", t: "ROIC vs cost of capital" },
        { s: "done", t: "Growth durability (5-yr)" },
        { s: "open", t: "Customer concentration > 10%?" },
      ],
      docs: ["Form 10 p.40", "Investor Day"],
      next: "Pull customer-concentration breakdown from Form 10 §3.",
      nextHas: true,
    },
    {
      name: "Capital", letter: "C", score: 45, color: "r",
      qs: [
        { s: "done", t: "Post-spin leverage 3.1x" },
        { s: "blocked", t: "Pension liability funding" },
        { s: "open", t: "Working capital needs" },
      ],
      docs: ["Form 10 p.87", "—pension footnote pending—"],
      next: "Pension footnote (expected today) — blocking 2 questions.",
      nextHas: true,
      isFocus: true,
    },
    {
      name: "Valuation", letter: "V", score: 60, color: "a",
      qs: [
        { s: "done", t: "EV/EBITDA vs peers" },
        { s: "open", t: "Sum-of-parts vs market cap" },
        { s: "open", t: "Implied free cash flow yield" },
      ],
      docs: ["Investor Day", "Peer comps"],
      next: "Build SoTP — peer multiples already loaded.",
      nextHas: true,
    },
    {
      name: "Incentives", letter: "I", score: 48, color: "r",
      qs: [
        { s: "done", t: "CEO equity stake at spin" },
        { s: "blocked", t: "Comp plan KPIs" },
        { s: "open", t: "Board independence" },
      ],
      docs: ["DEF 14A pending"],
      next: "Fetch DEF 14A from EDGAR.",
      nextHas: true,
    },
  ];

  const docs = [
    { name: "Form 10", meta: "312 pages · EDGAR", activates: ["S","B","C","V"], status: "Indexed" },
    { name: "Parent 10-K (CTL)", meta: "218 pages · EDGAR", activates: ["S","B"], status: "Indexed" },
    { name: "Investor Day deck", meta: "Uploaded May 14", activates: ["B","V"], status: "Indexed" },
    { name: "Pension footnote", meta: "Expected today", activates: ["C"], status: "Pending", pending: true },
    { name: "DEF 14A", meta: "Not yet on EDGAR", activates: ["I"], status: "Missing", pending: true },
  ];

  return (
    <div className="altA">
      <div className="altA-head">
        <div>
          <h1 className="h1">LUMN <span className="muted">· spinoff from CenturyTel</span></h1>
          <div className="sub">12 of 18 sub-questions answered · 3 blocked on missing documents</div>
        </div>
        <div>
          <div className="muted" style={{ fontSize: 12, textAlign: "right", marginBottom: 2 }}>COMPOSITE</div>
          <div className="composite"><span className="big">63</span><span className="denom">/ 100</span></div>
        </div>
      </div>

      <div className="altA-guide">
        <span style={{ color: "var(--blue)" }}>{I.arrowR}</span>
        <div>
          <div className="label">Next best action</div>
          <div className="what">Capital is your weakest dimension and 1 question is blocked by the pension footnote (expected today).</div>
          <div className="why">Unblocking this lifts confidence on 2 sub-questions and likely moves the composite +6 to +12.</div>
        </div>
        <button className="btn primary">Open Capital {I.arrowR}</button>
      </div>

      <div className="altA-dims">
        {dims.map(d => (
          <div key={d.name}
            className={"dim-card " + (focus === d.name ? "focus" : "")}
            onClick={() => setFocus(d.name)}>
            <div className="top">
              <div className="row" style={{ gap: 8 }}>
                <span className={"letter " + d.color}
                      style={{
                        background: d.color === "g" ? "var(--green-soft)" : d.color === "a" ? "var(--amber-soft)" : "var(--red-soft)",
                        color: d.color === "g" ? "var(--green)" : d.color === "a" ? "var(--amber)" : "var(--red)",
                      }}>{d.letter}</span>
                <span className="name">{d.name}</span>
              </div>
              <div className={"score " + (d.scoreText ? "text" : "")}>{d.score}</div>
            </div>
            <div className="conf-bar"><span className={d.color}
              style={{ width: typeof d.score === "number" ? d.score + "%" : "100%" }} /></div>
            <div className="checklist">
              {d.qs.map((q, i) => (
                <div key={i} className={"qrow " + q.s}>
                  <span className="pip" />
                  <span>{q.t}</span>
                </div>
              ))}
            </div>
            <div className="next">
              <div><b>Next:</b> {d.next}</div>
              {d.nextHas && (
                <div className="docs">
                  {d.docs.map((dc, i) => <span key={i} className="doc-tag">{dc}</span>)}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="altA-docs">
        <div className="h">
          <span>Documents in play · 5</span>
          <div className="actions">
            <button className="btn">{I.plus} Add document</button>
          </div>
        </div>
        {docs.map((d, i) => (
          <div key={i} className={"altA-doc-row " + (d.pending ? "pending" : "")}>
            <span style={{ color: d.pending ? "var(--amber)" : "var(--text-3)" }}>{d.pending ? I.clock : I.doc}</span>
            <div>
              <div className="name">{d.name}</div>
              <div className="meta">{d.meta}</div>
            </div>
            <div>
              <div className="activates-label" style={{ marginBottom: 4 }}>Activates</div>
              <div className="dim-chips">
                {["S","B","C","V","I"].map(L => (
                  <span key={L} className={"dim-chip " + (d.activates.includes(L) ? "active" : "")}>{L}</span>
                ))}
              </div>
            </div>
            <div className="status-tag">{d.status}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ===========================================================
   ALT B · "Evidence Board"
   Two-pane. Left: docs with activation chips. Right: dimensions
   that expand to show sub-questions and citations.
=========================================================== */
function AltB() {
  const [selectedDoc, setSelectedDoc] = React.useState("Form 10");
  const [expanded, setExpanded] = React.useState("Capital");

  const docs = [
    { name: "Form 10", meta: "312 pages · indexed May 2", activates: ["S","B","C","V"] },
    { name: "Parent 10-K (CTL)", meta: "218 pages · indexed May 2", activates: ["S","B"] },
    { name: "Investor Day deck", meta: "Uploaded · 47 pages", activates: ["B","V"] },
    { name: "Q1 earnings transcript", meta: "Indexed · 31 pages", activates: ["B","C"] },
    { name: "Pension footnote", meta: "Expected today · blocking Capital", activates: ["C"], pending: true },
    { name: "DEF 14A", meta: "Not yet on EDGAR", activates: ["I"], pending: true },
  ];

  const dims = [
    {
      name: "Setup", score: "Played", scoreText: true, complete: 3, total: 3, color: "g",
      qs: [
        { s: "done", q: "Will the spin be excluded from CTL's indices?", a: "Yes — excluded from 3 (S&P 500, Russell 1000, MSCI USA). Forced selling estimated $410M.", cite: "Form 10 p.12" },
        { s: "done", q: "Cap below institutional minimum?", a: "$1.6B — excluded from 2 of 4 typical large-cap mandates.", cite: "Peer comps" },
      ],
    },
    {
      name: "Business quality", score: 82, complete: 4, total: 5, color: "g",
      qs: [
        { s: "done", q: "ROIC vs cost of capital?", a: "18.4% ROIC vs 9.2% WACC — top-quartile peer set.", cite: "Form 10 p.40" },
        { s: "open", q: "Customer concentration above 10%?", a: null, cite: null },
      ],
    },
    {
      name: "Capital structure", score: 45, complete: 1, total: 3, color: "r", expand: true,
      qs: [
        { s: "done", q: "Post-spin net debt and leverage?", a: "$1.2B net debt · 3.1x EBITDA · investment grade BB+.", cite: "Form 10 p.85" },
        { s: "blocked", q: "Pension liability — funding shortfall?", a: "$340M liability disclosed. CFO declined to quantify shortfall on Q1 call.", cite: "Form 10 p.87 · Q1 call" },
        { s: "open", q: "Working capital needs at spin?", a: null, cite: null },
      ],
    },
    { name: "Valuation", score: 60, complete: 2, total: 4, color: "a", qs: [] },
    { name: "Incentives", score: 48, complete: 1, total: 3, color: "r", qs: [] },
  ];

  return (
    <div className="altB">
      <div className="altB-head">
        <div>
          <h1 className="h1">LUMN spinoff · evidence board</h1>
          <div className="sub">Composite 63 / 100 · click any document to highlight what it backs.</div>
        </div>
        <div className="row" style={{ gap: 8 }}>
          <button className="btn">{I.plus} Add document</button>
          <button className="btn primary">View memo {I.arrowR}</button>
        </div>
      </div>

      <div className="altB-banner">
        <span style={{ color: "var(--amber)" }}>{I.alert}</span>
        <div>
          <div className="label">Recommended next</div>
          <div className="what">Capital is at 1/3 questions answered, with pension footnote blocking 1. Unblocking lifts composite ~+8.</div>
        </div>
        <button className="btn">Focus Capital {I.arrowR}</button>
      </div>

      <div className="altB-grid">
        <div className="altB-docs">
          <div className="head">
            <span className="t">Documents · 6</span>
            <button className="btn ghost" style={{ padding: "6px 10px", fontSize: 13 }}>{I.plus} Add</button>
          </div>
          {docs.map(d => (
            <div key={d.name}
              className={"altB-doc " + (selectedDoc === d.name ? "selected " : "") + (d.pending ? "pending" : "")}
              onClick={() => setSelectedDoc(d.name)}>
              <div className="row1">
                <span style={{ color: d.pending ? "var(--amber)" : "var(--text-3)" }}>{d.pending ? I.clock : I.doc}</span>
                <span className="name">{d.name}</span>
                <div className="dim-chips">
                  {["S","B","C","V","I"].map(L => (
                    <span key={L} className={"dim-chip " + (d.activates.includes(L) ? "active" : "")}>{L}</span>
                  ))}
                </div>
              </div>
              <div className="meta">{d.meta}</div>
            </div>
          ))}
        </div>

        <div className="altB-dims">
          {dims.map(d => {
            const open = expanded === d.name;
            return (
              <div key={d.name} className={"altB-dim " + (open ? "expanded" : "")}>
                <div className="row" onClick={() => setExpanded(open ? "" : d.name)}>
                  <span className={"dim-chip " + d.color}>{d.name[0]}</span>
                  <div>
                    <div className="name">{d.name}</div>
                    <div className="progress">{d.complete} of {d.total} questions answered</div>
                  </div>
                  <div className={"score " + (d.scoreText ? "text" : "")}>{d.score}</div>
                  <span style={{ color: "var(--text-3)" }}>{open ? I.up : I.down}</span>
                </div>
                {open && d.qs.length > 0 && (
                  <div className="qbody">
                    {d.qs.map((q, i) => (
                      <div key={i} className={"altB-q " + q.s}>
                        <span className="pip" />
                        <div className="q">
                          {q.q}
                          {q.a && <span className="a">{q.a}</span>}
                          {!q.a && <span className="a muted" style={{ color: "var(--text-3)" }}>— not yet answered —</span>}
                        </div>
                        {q.cite ? <span className="cite">{q.cite}</span> : <button className="btn" style={{ padding: "4px 10px", fontSize: 12 }}>Answer</button>}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ===========================================================
   ALT C · "Guided co-pilot"
   Conversational. System asks next best question, you answer
   with a citation, score updates live on the side.
=========================================================== */
function AltC() {
  const turns = [
    { who: "sys", meta: "Setup · Q2 of 3", body: "Will the spin be excluded from CTL's indices?", large: true },
    { who: "you", body: "Yes — Form 10 p.12 lists exclusion from S&P 500, Russell 1000, and MSCI USA. Forced selling ~$410M.", cites: ["Form 10 p.12"] },
    { who: "sys", body: "Got it. Setup confidence is now strong on the index angle.",
      delta: <>Score <b style={{ color: "var(--text)" }}>Setup</b> <span className="up">72 → 88</span></>,
      followUp: "Next, I need to score Capital — but the pension footnote is missing and it blocks 2 questions. Want to skip Capital for now or chase the doc?",
      actions: ["Chase pension footnote", "Skip Capital for now", "Ask something else"],
    },
  ];

  const dims = [
    { name: "Setup",      score: 88, color: "g", meta: "3 of 3 answered" },
    { name: "Business",   score: 82, color: "g", meta: "4 of 5 answered" },
    { name: "Capital",    score: 45, color: "r", meta: "1 of 3 · 1 blocked", current: true },
    { name: "Valuation",  score: 60, color: "a", meta: "2 of 4 answered" },
    { name: "Incentives", score: 48, color: "r", meta: "1 of 3 · 1 blocked" },
  ];
  const composite = 66;
  const openQs = [
    { t: "Pension funding shortfall", state: "blocked" },
    { t: "Working capital needs at spin", state: "open" },
    { t: "Customer concentration > 10%", state: "open" },
    { t: "Sum-of-parts vs market cap", state: "open" },
    { t: "Comp plan KPIs", state: "blocked" },
    { t: "Board independence", state: "open" },
  ];

  return (
    <div className="altC">
      <div className="altC-head">
        <div>
          <h1 className="h1">LUMN spinoff · guided</h1>
          <div className="sub">I'll ask the next-best question. Answer or ask your own.</div>
        </div>
        <div className="progress-text">12 of 18 questions answered</div>
      </div>
      <div className="altC-progress"><span style={{ width: (12/18*100) + "%" }} /></div>

      <div className="altC-grid">
        <div className="altC-feed">
          {turns.map((t, i) => (
            <div key={i} className="altC-turn">
              <span className={"altC-avatar " + t.who}>{t.who === "sys" ? "M" : "You"}</span>
              <div className={"altC-bubble " + t.who}>
                {t.meta && <div className="meta">{t.meta}</div>}
                <div className={"body " + (t.large ? "large" : "")}>{t.body}</div>
                {t.cites && (
                  <div className="cites">
                    {t.cites.map((c, j) => <span key={j} className="cite">{c}</span>)}
                  </div>
                )}
                {t.delta && <div className="delta-line">{t.delta}</div>}
                {t.followUp && <div className="body" style={{ marginTop: 12 }}>{t.followUp}</div>}
                {t.actions && (
                  <div className="actions">
                    {t.actions.map((a, j) =>
                      <button key={j} className={"btn " + (j === 0 ? "primary" : "")}>{a}</button>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          <div className="altC-input">
            <input placeholder="Answer, ask a question, or paste a citation…" />
            <button className="btn primary">Send {I.arrowR}</button>
          </div>
        </div>

        <div className="altC-side">
          <div className="h">COMPOSITE</div>
          <div className="composite"><span className="big">{composite}</span><span className="denom">/ 100</span></div>

          {dims.map(d => (
            <div key={d.name} className={"altC-side-dim " + (d.current ? "current" : "")}>
              <div className="name">
                <span className={"dim-chip " + d.color} style={{ marginRight: 8, width: 18, height: 18, fontSize: 10.5 }}>{d.name[0]}</span>
                {d.name}
              </div>
              <div className="val">{d.score}</div>
              <div className="row2">
                <div className="conf-bar" style={{ marginTop: 6 }}>
                  <span className={d.color} style={{ width: d.score + "%" }} />
                </div>
                <div className="meta">{d.meta}{d.current && " · in focus"}</div>
              </div>
            </div>
          ))}

          <div className="open-list">
            <div className="h" style={{ margin: 0, marginBottom: 6 }}>OPEN QUESTIONS · 6</div>
            {openQs.map((q, i) => (
              <div key={i} className="row">
                <span className={"pip " + q.state} />
                <span>{q.t}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

window.Alternatives = { AltA, AltB, AltC };

/* global React, SpinoffUI, MeridianData */
const { I } = SpinoffUI;
const { DIMS, CRITERIA, DIM_STATE, COMPOSITE, DOCS } = MeridianData;

/* ====================================================================
   Dimension card — collapsed view (top 3 of 7 criteria + next action)
==================================================================== */
function DimCard({ dim, focused, onFocus }) {
  const state = DIM_STATE[dim.id];
  const crits = CRITERIA[dim.id];
  const counts = crits.reduce((acc, c) => { acc[c.status] = (acc[c.status] || 0) + 1; return acc; }, {});
  const done = counts.done || 0;
  const open = (counts.open || 0);
  const blocked = (counts.blocked || 0);
  const partial = (counts.partial || 0);
  const total = crits.length;
  const scoreCls = typeof state.score === "number"
    ? (state.score >= 80 ? "score-green" : state.score >= 60 ? "score-amber" : "score-red")
    : "";
  // Next-best criterion: prefer blocked > open > partial
  const next = crits.find(c => c.status === "blocked")
            || crits.find(c => c.status === "open")
            || crits.find(c => c.status === "partial");

  // Top 4 criteria preview
  const preview = crits.slice(0, 4);

  return (
    <div className={"m-dim " + dim.cls + (focused ? " focus" : "")} onClick={() => onFocus(dim.id)}>
      <div className="top">
        <div className="name-row">
          <span className="letter">{dim.letter}</span>
          <span className="name">{dim.name}</span>
        </div>
        <div className={"score " + (state.playedOut ? "text " : "") + scoreCls}>
          {state.playedOut ? "Played" : state.score}
        </div>
      </div>

      <div>
        <div className="coverage-label">
          <span>Coverage</span>
          <span>{done}/{total}{blocked ? ` · ${blocked} blocked` : ""}</span>
        </div>
        <div className="coverage-bar" style={{ marginTop: 4 }}>
          {crits.map((c, i) => (
            <span key={i} className={"pip " + (
              c.status === "done" ? "full" :
              c.status === "partial" ? "partial" :
              c.status === "blocked" ? "blocked" : ""
            )} />
          ))}
        </div>
      </div>

      <div className="checklist">
        {preview.map((c, i) => (
          <div key={i} className={"qrow " + c.status}>
            <span className="pip" />
            <span>{c.name}</span>
          </div>
        ))}
        {crits.length > preview.length && (
          <div className="more" style={{ marginLeft: 22 }}>+ {crits.length - preview.length} more criteria</div>
        )}
      </div>

      <div className="next-block">
        {next ? (
          <>
            <div className="next-line"><b>Next:</b> {next.name}</div>
            <div className={"next-doc " + (next.docState === "pending" || next.docState === "missing" ? "pending" : "")}>
              {next.docState === "pending" || next.docState === "missing"
                ? <>{I.clock}<span>{next.doc} · {next.docState}</span></>
                : <>{I.doc}<span>{next.doc}</span></>}
            </div>
          </>
        ) : (
          <div className="next-line" style={{ color: "var(--green)" }}>✓ All criteria answered</div>
        )}
      </div>
    </div>
  );
}

/* ====================================================================
   Expanded dimension panel — full 7-criterion deep dive
==================================================================== */
function DimDetail({ dimId, onClose }) {
  const dim = DIMS.find(d => d.id === dimId);
  const crits = CRITERIA[dimId];
  const state = DIM_STATE[dimId];
  const scoreCls = typeof state.score === "number"
    ? (state.score >= 80 ? "score-green" : state.score >= 60 ? "score-amber" : "score-red")
    : "";

  // Sample agent commentary — would come from latest committee run
  const agentNote = {
    setup:      { who: "Setup Specialist · last run May 14", body: "$1.6B cap excludes 3 indices CTL sits in; forced selling well-quantified. Setup is played." },
    business:   { who: "Business Quality Analyst · last run May 14", body: "ROIC 18.4% top-quartile; FCF conversion 92%. One open question on customer concentration weighs lightly." },
    capital:    { who: "Devil's Advocate · last run May 14", body: "Pension transfer materially under-disclosed. ~$90M extra funding implied. Score will not stabilize until footnote lands." },
    valuation:  { who: "Valuation Analyst · last run May 14", body: "Multiples attractive vs. peers but SoTP missing. Build it from Investor Day projections — peer multiples already loaded." },
    incentives: { who: "Devil's Advocate · last run May 14", body: "DEF 14A not yet on EDGAR; comp structure and option vesting unknown. Two criteria will remain blocked until post-distribution." },
  }[dimId];

  return (
    <div className={"m-dim-detail " + dim.cls}>
      <div className={"head " + dim.cls}>
        <span className="letter">{dim.letter}</span>
        <div>
          <div className="name">{dim.name}</div>
          <div className="keyq">{dim.key}</div>
        </div>
        <div className={"score " + (state.playedOut ? "text " : "") + scoreCls}>
          {state.playedOut ? "Played" : state.score}
        </div>
        <span className="muted" style={{ fontSize: 12.5 }}>Confidence: {state.conf}</span>
        <button className="close" onClick={onClose}>Collapse ▲</button>
      </div>

      {agentNote && (
        <div className="agent-note">
          <span className="ava">M</span>
          <div>
            <div className="who">{agentNote.who}</div>
            <div className="body">{agentNote.body}</div>
          </div>
        </div>
      )}

      <div>
        {crits.map((c, i) => (
          <div key={i} className="m-crit-row">
            <span className={"status-pip " + c.status} />
            <div className="name">
              {c.name}
              {c.note && <div className="sub">{c.note}</div>}
            </div>
            <div className={"value " + (c.value ? "" : "muted")}>
              {c.value || "— not yet extracted —"}
            </div>
            <div className={"doc-chip " + (c.docState === "pending" ? "pending" : c.docState === "missing" ? "miss" : "")}>
              {c.docState === "pending" ? I.clock : c.docState === "missing" ? I.alert : I.doc}
              <span>{c.doc}</span>
            </div>
            <div>
              {c.status === "blocked" ? (
                <button className="btn" style={{ padding: "5px 10px", fontSize: 12 }}>Chase doc {I.arrowUR}</button>
              ) : c.status === "open" ? (
                <button className="btn" style={{ padding: "5px 10px", fontSize: 12 }}>Extract</button>
              ) : (
                <button className="btn ghost" style={{ padding: "5px 10px", fontSize: 12 }}>View cite</button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ====================================================================
   Workspace header (composite score + state + primary actions)
==================================================================== */
function WorkspaceHeader({ onRunCommittee, committeeRunning }) {
  return (
    <div className="m-hero">
      <div className="composite-ring" style={{ "--p": (COMPOSITE * 3.6) + "deg" }}>
        <span className="v">{COMPOSITE}</span>
      </div>
      <div className="meta">
        <div className="label">LUMN · spinoff from CenturyTel · CIK 0000018926</div>
        <div className="name">Lumen Spinco, Inc. <span className="muted" style={{ fontWeight: 400 }}>· Telecom services</span></div>
        <div className="sub">Composite 65 / 100 · 26 of 35 criteria answered · 4 blocked</div>
      </div>
      <div className="stats">
        <div className="stat">
          <div className="l">Status</div>
          <div className="v text">
            <span className="m-state researching"><span className="dot" />Researching</span>
          </div>
        </div>
        <div className="stat">
          <div className="l">Last committee</div>
          <div className="v text">May 14 · v2</div>
        </div>
        <div className="stat">
          <div className="l">Memo</div>
          <div className="v text amber">v2 · stale</div>
        </div>
      </div>
    </div>
  );
}

/* ====================================================================
   Committee run output panel (slides in below dim grid when run)
==================================================================== */
function CommitteePanel({ onClose }) {
  const agents = [
    { k: "S", color: "var(--d-setup)", soft: "var(--d-setup-soft)", who: "Setup Specialist", body: "Forced-selling magnitude unchanged. Index exclusions confirmed by post-spin float estimate. Score holds at 88.", delta: "→ 88", deltaCls: "flat" },
    { k: "B", color: "var(--d-business)", soft: "var(--d-business-soft)", who: "Business Quality", body: "ROIC and FCF conversion top-quartile. Light penalty for unanswered customer concentration. 82 → 82.", delta: "→ 82", deltaCls: "flat" },
    { k: "C", color: "var(--d-capital)", soft: "var(--d-capital-soft)", who: "Capital Structure", body: "Capital cannot stabilize until pension footnote lands. Provisional score weighted toward Devil's Advocate.", delta: "60 → 45", deltaCls: "down" },
    { k: "V", color: "var(--d-valuation)", soft: "var(--d-valuation-soft)", who: "Valuation", body: "Multiples constructive vs. peers (EV/EBIT 9.4x median 11.8x). SoTP still missing — capped at 60.", delta: "→ 60", deltaCls: "flat" },
    { k: "I", color: "var(--d-incentives)", soft: "var(--d-incentives-soft)", who: "Incentives", body: "Two blocked criteria from missing DEF 14A. Buyback authorization is a positive prior. Score 48.", delta: "→ 48", deltaCls: "flat" },
    { k: "D", color: "var(--red)", soft: "var(--red-soft)", who: "Devil's Advocate", body: "Pension transfer under-disclosed. ~$90M extra funding implied. Recommend sized entry (5%) pending May 23 footnote.", delta: "Composite 65", deltaCls: "flat" },
  ];
  return (
    <div className="m-comm-panel">
      <div className="head">
        <span className="t"><span className="pulse" />Committee · v3 — round 3 of 3</span>
        <div className="m-actions">
          <button className="btn">Pause</button>
          <button className="btn">Generate memo {I.arrowR}</button>
          <button className="btn ghost" onClick={onClose}>Close</button>
        </div>
      </div>
      <div className="agents">
        {agents.map((a, i) => (
          <div className="arow" key={i}>
            <span className="ava" style={{ background: a.soft, color: a.color }}>{a.k}</span>
            <div>
              <div className="who">{a.who}</div>
              <div className="body">{a.body}</div>
            </div>
            <div className={"delta " + a.deltaCls}>{a.delta}</div>
            <button className="btn ghost" style={{ padding: "4px 10px", fontSize: 12 }}>Trace</button>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ====================================================================
   Documents pane (right rail)
==================================================================== */
function DocsPane() {
  return (
    <div className="m-docs-pane">
      <div className="head">
        <span className="t">Documents · {DOCS.length}</span>
        <button className="btn" style={{ padding: "6px 10px", fontSize: 12.5 }}>{I.plus} Add</button>
      </div>
      {DOCS.map(d => (
        <div key={d.id} className={"m-doc-row " + (d.state !== "indexed" ? "pending" : "")}>
          <span style={{ color: d.state === "indexed" ? "var(--text-3)" : "var(--amber)" }}>
            {d.state === "indexed" ? I.doc : I.clock}
          </span>
          <div>
            <div className="name">{d.name}</div>
            <div className="meta">
              {d.type}{d.pages ? ` · ${d.pages} pp` : ""}{d.method !== "—" ? ` · ${d.method}` : ""}
              {d.note ? ` · ${d.note}` : ""}
            </div>
          </div>
          <div />
          <div>
            <div className="activates-label">Activates</div>
            <div className="activates dim-chips">
              {["S","B","C","V","I"].map(L => (
                <span key={L} className={"dim-chip " + (d.unlocks.includes(L) ? "active" : "")}>{L}</span>
              ))}
            </div>
          </div>
          <div className="muted" style={{ fontSize: 12, textAlign: "right", textTransform: "capitalize" }}>{d.state}</div>
        </div>
      ))}
    </div>
  );
}

/* ====================================================================
   Activity feed — memos, committee, decisions, promises, docs, updates
==================================================================== */
function ActivityFeed() {
  const [filter, setFilter] = React.useState("All");
  const filters = ["All", "Memos", "Committee", "Decisions", "Promises", "Docs"];

  const events = [
    { type: "doc", title: "Pension footnote ingest pending", meta: "expected today (May 23)", time: "—",
      body: "Capital dimension has 1 blocked criterion until this lands." },
    { type: "promise", title: "Mgmt promised SoTP at Q1 investor day", meta: "Q1 transcript p.14", time: "May 6",
      body: "No date committed. Coach: log as a tracked promise." },
    { type: "doc", title: "8-K filed · $200M buyback authorized", meta: "EDGAR · auto-ingested", time: "May 2",
      body: "Incentives criterion 3 changed: open → done." },
    { type: "committee", title: "Committee v2 ran — composite 60", meta: "5 agents · 3 rounds · 4m12s", time: "May 14",
      body: "Capital dragged down by unresolved pension liability." },
    { type: "memo", title: "Memo v2 generated by Opus 4.7", meta: "8 sections · 2,140 tokens", time: "May 14",
      body: "Watch — pending pension disclosure" },
    { type: "decision", title: "Logged: Watch · conviction 6", meta: "Driver: Setup played; Risk: pension shortfall", time: "May 14",
      body: "Re-evaluate when footnote lands." },
  ];
  const ico = {
    memo: I.copy2, committee: I.user, decision: I.check, promise: I.bookmark, doc: I.doc, update: I.bell,
  };
  const filt = filter === "All" ? events : events.filter(e =>
    (filter === "Memos" && e.type === "memo") ||
    (filter === "Committee" && e.type === "committee") ||
    (filter === "Decisions" && e.type === "decision") ||
    (filter === "Promises" && e.type === "promise") ||
    (filter === "Docs" && e.type === "doc"));

  return (
    <div className="m-feed">
      <div className="head">
        <span className="t">Activity</span>
        <div className="filter">
          {filters.map(f => (
            <span key={f} className={"chip " + (filter === f ? "active" : "")} onClick={() => setFilter(f)}>{f}</span>
          ))}
        </div>
      </div>
      {filt.map((e, i) => (
        <div key={i} className={"m-feed-row event-" + e.type}>
          <span className="ico-wrap">{ico[e.type]}</span>
          <div>
            <div className="title">{e.title}</div>
            <div className="meta">{e.meta}</div>
            <div className="body">{e.body}</div>
          </div>
          <div className="time">{e.time}</div>
        </div>
      ))}
    </div>
  );
}

window.MeridianWorkspaceParts = { DimCard, DimDetail, WorkspaceHeader, CommitteePanel, DocsPane, ActivityFeed, QuarterlyReviewPanel };

/* ====================================================================
   Quarterly Review panel — opens when a new 10-Q is ingested
==================================================================== */
function QuarterlyReviewPanel({ onClose, onRunCommittee }) {
  const changes = [
    { crit: "Net debt",            dim: "capital",   old: "$1.2B",      next: "$1.35B",    dir: "down",  doc: "Q1 10-Q · p.42" },
    { crit: "Net debt / EBITDA",   dim: "capital",   old: "3.1x",       next: "3.4x",      dir: "down",  doc: "Q1 10-Q · p.42" },
    { crit: "Pension / OPEB",      dim: "capital",   old: "blocked",    next: "$430M",     dir: "down",  doc: "Q1 10-Q note 14" },
    { crit: "Gross margin trend",  dim: "business",  old: "−40 bps",    next: "−95 bps",   dir: "down",  doc: "XBRL · Q1" },
    { crit: "FCF conversion",      dim: "business",  old: "92%",        next: "88%",       dir: "down",  doc: "XBRL · Q1" },
    { crit: "Capex intensity",     dim: "capital",   old: "5.2%",       next: "5.4%",      dir: "down",  doc: "XBRL · Q1" },
  ];
  const promises = [
    { state: "missed", what: "SoTP framework by Q1 investor day",   src: "Q4 2025 transcript p.14",  outcome: "Not mentioned in Q1 prepared remarks." },
    { state: "kept",   what: "$200M buyback authorization announced", src: "8-K · May 2, 2026",       outcome: "Authorized at upper end of prior guidance." },
    { state: "missed", what: "Pension funding plan disclosed",       src: "Q1 2026 call transcript", outcome: "CFO again declined to quantify on Q1 call." },
  ];
  const dimColor = {
    setup: "var(--d-setup)", business: "var(--d-business)", capital: "var(--d-capital)",
    valuation: "var(--d-valuation)", incentives: "var(--d-incentives)",
  };
  return (
    <div className="m-qr-panel">
      <div className="head">
        <span className="t"><span className="dot" />Quarterly review · Q1 2026 10-Q ingested</span>
        <div className="m-actions">
          <button className="btn">Snooze</button>
          <button className="btn">Accept changes</button>
          <button className="btn pri" onClick={() => { onClose(); onRunCommittee && onRunCommittee(); }}>Re-run committee {I.arrowR}</button>
        </div>
      </div>

      <div className="pipeline">
        <span className="step"><span className="check">{I.check}</span> Filing fetched · EDGAR · 64 pp</span>
        <span className="step"><span className="check">{I.check}</span> Parsed · 18 sections</span>
        <span className="step"><span className="check">{I.check}</span> XBRL refreshed · 31 facts</span>
        <span className="step"><span className="check">{I.check}</span> Haiku sidecar · pension footnote extracted</span>
        <span className="step"><span className="check">{I.check}</span> Embeddings · 142 new chunks</span>
      </div>

      <div className="section">
        <div className="h">Criteria that moved · 6</div>
        {changes.map((c, i) => (
          <div key={i} className="m-qr-row">
            <span className="crit-dim-tag">
              <span className="dot" style={{ background: dimColor[c.dim] }} />
            </span>
            <div>
              <div style={{ fontWeight: 500 }}>{c.crit}</div>
              <div className="muted" style={{ fontSize: 11.5, marginTop: 2, textTransform: "capitalize" }}>{c.dim}</div>
            </div>
            <span className="old">{c.old}</span>
            <span className={"new " + c.dir}>→ {c.next}</span>
            <span className="doc-chip">{c.doc}</span>
          </div>
        ))}
      </div>

      <div className="section">
        <div className="h">Promises reconciled · 3</div>
        {promises.map((p, i) => (
          <div key={i} className={"m-qr-prom " + p.state}>
            <span className="pip" />
            <div>
              <div>{p.what}</div>
              <div className="src">{p.src} · {p.outcome}</div>
            </div>
            <span className="muted" style={{ fontSize: 11.5, textTransform: "uppercase", letterSpacing: 0.4 }}>{p.state}</span>
          </div>
        ))}
      </div>

      <div className="section">
        <div className="h">Inferred composite</div>
        <div className="row" style={{ alignItems: "baseline", gap: 14 }}>
          <span style={{ fontSize: 32, fontWeight: 500, fontVariantNumeric: "tabular-nums" }}>65 → <span style={{ color: "var(--red)" }}>56</span></span>
          <span className="muted" style={{ fontSize: 13 }}>Capital and Business each lose ground. Committee should be re-run to confirm.</span>
        </div>
      </div>
    </div>
  );
}

window.MeridianWorkspaceParts = { DimCard, DimDetail, WorkspaceHeader, CommitteePanel, DocsPane, ActivityFeed, QuarterlyReviewPanel };

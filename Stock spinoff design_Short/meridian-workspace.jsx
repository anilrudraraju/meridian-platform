/* global React, SpinoffUI, MeridianData */
const { I } = SpinoffUI;
const { DIMS, CRITERIA, DIM_STATE, COMPOSITE, DOCS, SETUP_META, VALUATION_COMPS, DECISION } = MeridianData;

/* ====================================================================
   Dimension card — collapsed view (top 3 of 7 criteria + next action)
==================================================================== */
function DimCard({ dim, focused, onFocus, scenario }) {
  const base = DIM_STATE[dim.id];
  const over = scenario && scenario.dims && scenario.dims[dim.id];
  const state = over ? { ...base, ...over } : base;
  const crits = CRITERIA[dim.id];
  const counts = crits.reduce((acc, c) => { acc[c.status] = (acc[c.status] || 0) + 1; return acc; }, {});
  let done = counts.done || 0;
  let open = (counts.open || 0);
  let blocked = (counts.blocked || 0);
  let partial = (counts.partial || 0);
  const total = crits.length;
  if (over && over.coverage) {
    done = over.coverage.done;
    blocked = over.coverage.blocked;
    /* keep partial/open derived only for visual pip variety */
  }
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
        <div className={"score " + scoreCls}>
          {state.score}
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
    </div>
  );
}

/* ====================================================================
   Setup metadata panel — situation type, spin type, selling tracker
   Rendered at the top of DimDetail for dimId === "setup"
==================================================================== */
function SetupMetaPanel() {
  const meta    = SETUP_META;
  const tracker = meta.selling_tracker;
  const pctOfThreshold = Math.min(100, Math.round((tracker.pct_traded / 40) * 100));

  const situationTypes = ["Demerger Arbitrage", "Indiscriminate Selling", "Cheap Absolute+Relative", "Dividend Catalyst"];
  const spinoffTypes   = ["100% Spin-off", "Partial Spin-off", "Equity Carve-out", "Split-off / Exchange Offer", "Reverse Morris Trust"];

  return (
    <div style={{ marginBottom: 20 }}>
      {/* Situation type pills */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: "#a4a4ad", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8 }}>Situation Type</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {situationTypes.map(t => {
            const active = t === meta.situation_type;
            return (
              <span key={t} style={{
                padding: "5px 12px", borderRadius: 6, fontSize: 12.5,
                fontWeight: active ? 600 : 400,
                background: active ? "var(--d-setup-soft)" : "rgba(255,255,255,0.04)",
                color: active ? "var(--d-setup)" : "var(--text-3)",
                border: active ? "1px solid var(--d-setup)" : "1px solid rgba(255,255,255,0.06)",
                cursor: "pointer",
              }}>{t}</span>
            );
          })}
        </div>
      </div>

      {/* Spin-off structure pills */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: "#a4a4ad", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8 }}>Spin-off Structure</div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {spinoffTypes.map(t => {
            const active = t === meta.spinoff_type;
            return (
              <span key={t} style={{
                padding: "5px 12px", borderRadius: 6, fontSize: 12.5,
                fontWeight: active ? 600 : 400,
                background: active ? "rgba(255,255,255,0.08)" : "rgba(255,255,255,0.03)",
                color: active ? "var(--text)" : "var(--text-3)",
                border: active ? "1px solid rgba(255,255,255,0.18)" : "1px solid rgba(255,255,255,0.06)",
                cursor: "pointer",
              }}>{t}</span>
            );
          })}
        </div>
      </div>

      {/* Indiscriminate Selling tracker — only shown when that setup type is active */}
      {meta.situation_type === "Indiscriminate Selling" && (
        <div style={{
          background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 10, padding: "14px 16px",
        }}>
          <div style={{ fontSize: 11, fontWeight: 600, color: "#a4a4ad", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 14 }}>Indiscriminate Selling Tracker</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 16, marginBottom: 16 }}>
            <div>
              <div style={{ fontSize: 11, color: "var(--text-3)", marginBottom: 3 }}>Days since spin</div>
              <div style={{ fontSize: 22, fontWeight: 600, color: tracker.days_since_spin <= 9 ? "var(--amber)" : "var(--green)" }}>{tracker.days_since_spin}</div>
              <div style={{ fontSize: 11, color: "var(--text-3)" }}>avg bottom ~9 days</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "var(--text-3)", marginBottom: 3 }}>Cumulative volume</div>
              <div style={{ fontSize: 22, fontWeight: 600 }}>{tracker.cumulative_volume_m}M</div>
              <div style={{ fontSize: 11, color: "var(--text-3)" }}>of {tracker.shares_outstanding_m}M shares</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "var(--text-3)", marginBottom: 3 }}>% shares traded</div>
              <div style={{ fontSize: 22, fontWeight: 600, color: tracker.pct_traded >= 40 ? "var(--green)" : "var(--amber)" }}>{tracker.pct_traded}%</div>
              <div style={{ fontSize: 11, color: "var(--text-3)" }}>threshold ~40%</div>
            </div>
            <div>
              <div style={{ fontSize: 11, color: "var(--text-3)", marginBottom: 3 }}>Price trend</div>
              <div style={{ fontSize: 22, fontWeight: 600, color: tracker.price_trend === "stabilizing" ? "var(--amber)" : tracker.price_trend === "recovering" ? "var(--green)" : "var(--red)", textTransform: "capitalize" }}>{tracker.price_trend}</div>
              <div style={{ fontSize: 11, color: "var(--text-3)" }}>avg daily {tracker.avg_daily_vol_m}M</div>
            </div>
          </div>
          <div style={{ marginBottom: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
              <span style={{ fontSize: 11, color: "var(--text-3)" }}>Selling exhaustion progress toward ~40% threshold</span>
              <span style={{ fontSize: 11, fontWeight: 600, color: tracker.pct_traded >= 40 ? "var(--green)" : "var(--amber)" }}>{pctOfThreshold}%</span>
            </div>
            <div style={{ height: 6, background: "rgba(255,255,255,0.08)", borderRadius: 3, overflow: "hidden" }}>
              <div style={{
                height: "100%", width: pctOfThreshold + "%",
                background: tracker.pct_traded >= 40 ? "var(--green)" : "var(--amber)",
                borderRadius: 3, transition: "width 0.4s ease",
              }} />
            </div>
          </div>
          <div style={{ fontSize: 12, color: "var(--text-3)", fontStyle: "italic" }}>{tracker.note}</div>
        </div>
      )}
    </div>
  );
}

/* ====================================================================
   Valuation comps table — rendered at the bottom of DimDetail for
   dimId === "valuation"
==================================================================== */
function ValuationCompsTable() {
  const c   = VALUATION_COMPS;
  const fmt = v => v != null ? v.toFixed(1) + "x" : "—";
  const cmp = (val, med) => {
    if (val == null || med == null) return "var(--text)";
    return val < med ? "var(--green)" : val > med * 1.1 ? "var(--red)" : "var(--text)";
  };
  const colTemplate = "130px 1fr 90px 90px 80px";
  const cellBase = { padding: "9px 14px", fontSize: 12.5, borderBottom: "1px solid rgba(255,255,255,0.04)" };
  const hdr = { ...cellBase, fontSize: 11, fontWeight: 600, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.4px", borderBottom: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.03)", padding: "8px 14px" };

  return (
    <div style={{ marginTop: 24 }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: "#a4a4ad", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 10 }}>Valuation vs. Peers</div>
      <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 8, overflow: "hidden" }}>
        {/* Header */}
        <div style={{ display: "grid", gridTemplateColumns: colTemplate }}>
          {["Ticker", "Company", "EV/EBIT", "EV/EBITDA", "P/FCF"].map((h, i) => (
            <div key={h} style={{ ...hdr, textAlign: i > 1 ? "right" : "left" }}>{h}</div>
          ))}
        </div>
        {/* Subject */}
        <div style={{ display: "grid", gridTemplateColumns: colTemplate, background: "rgba(255,255,255,0.04)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
          <div style={{ ...cellBase, fontWeight: 700, color: "var(--d-valuation)" }}>{c.subject.ticker}</div>
          <div style={{ ...cellBase, color: "var(--text-2)" }}>Subject company</div>
          <div style={{ ...cellBase, textAlign: "right", fontWeight: 700, color: cmp(c.subject.ev_ebit, c.sector_median.ev_ebit) }}>{fmt(c.subject.ev_ebit)}</div>
          <div style={{ ...cellBase, textAlign: "right", fontWeight: 700, color: cmp(c.subject.ev_ebitda, c.sector_median.ev_ebitda) }}>{fmt(c.subject.ev_ebitda)}</div>
          <div style={{ ...cellBase, textAlign: "right", fontWeight: 700, color: cmp(c.subject.p_fcf, c.sector_median.p_fcf) }}>{fmt(c.subject.p_fcf)}</div>
        </div>
        {/* Peers */}
        {c.peers.map((p, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: colTemplate }}>
            <div style={{ ...cellBase, fontWeight: 500 }}>{p.ticker}</div>
            <div style={{ ...cellBase, color: "var(--text-3)", fontSize: 12 }}>{p.name}</div>
            <div style={{ ...cellBase, textAlign: "right" }}>{fmt(p.ev_ebit)}</div>
            <div style={{ ...cellBase, textAlign: "right" }}>{fmt(p.ev_ebitda)}</div>
            <div style={{ ...cellBase, textAlign: "right" }}>{fmt(p.p_fcf)}</div>
          </div>
        ))}
        {/* Sector median */}
        <div style={{ display: "grid", gridTemplateColumns: colTemplate, background: "rgba(255,255,255,0.03)", borderTop: "1px solid rgba(255,255,255,0.10)" }}>
          <div style={{ ...cellBase, fontSize: 11, fontWeight: 700, color: "var(--text-3)", textTransform: "uppercase", borderBottom: "none" }}>Sector median</div>
          <div style={{ ...cellBase, borderBottom: "none" }} />
          <div style={{ ...cellBase, textAlign: "right", fontWeight: 600, color: "var(--text-2)", borderBottom: "none" }}>{fmt(c.sector_median.ev_ebit)}</div>
          <div style={{ ...cellBase, textAlign: "right", fontWeight: 600, color: "var(--text-2)", borderBottom: "none" }}>{fmt(c.sector_median.ev_ebitda)}</div>
          <div style={{ ...cellBase, textAlign: "right", fontWeight: 600, color: "var(--text-2)", borderBottom: "none" }}>{fmt(c.sector_median.p_fcf)}</div>
        </div>
      </div>
      <div style={{ fontSize: 11.5, color: "var(--text-3)", marginTop: 8 }}>Green = below sector median (cheaper). Source: XBRL + Form 10.</div>
    </div>
  );
}

/* ====================================================================
   Decision panel — conviction, position sizing, exit plan
   Rendered below the dim grid in the workspace "dims" tab
==================================================================== */
function DecisionPanel() {
  const d = DECISION;
  const statusColor = { "Pass": "var(--red)", "Watch": "var(--amber)", "Initiate": "var(--green)", "Full position": "var(--green)" }[d.status] || "var(--text)";

  return (
    <div style={{
      background: "var(--bg-elev)", border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 12, padding: "20px 24px", marginTop: 28,
    }}>
      {/* Title row */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 4 }}>Decision</div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 22, fontWeight: 700, color: statusColor }}>{d.status}</span>
            <span style={{ fontSize: 13, color: "var(--text-3)" }}>· composite {d.composite}/100 · {d.conviction.toLowerCase()} conviction</span>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn">Log decision</button>
          <button className="btn pri">Generate memo {I.arrowR}</button>
        </div>
      </div>

      {/* KPI grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 14, marginBottom: 16 }}>
        <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: "12px 14px" }}>
          <div style={{ fontSize: 11, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.4px", marginBottom: 6 }}>Suggested size</div>
          <div style={{ fontSize: 24, fontWeight: 700 }}>{d.suggested_position}</div>
          <div style={{ fontSize: 11.5, color: "var(--text-3)", marginTop: 4 }}>{d.conviction} conviction · Rich framework</div>
        </div>
        <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: "12px 14px" }}>
          <div style={{ fontSize: 11, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.4px", marginBottom: 6 }}>Setup</div>
          <div style={{ fontSize: 14, fontWeight: 600, lineHeight: 1.3 }}>{d.setup_type}</div>
          <div style={{ fontSize: 11.5, color: "var(--text-3)", marginTop: 4 }}>Selling ~{SETUP_META.selling_tracker.pct_traded}% exhausted</div>
        </div>
        <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: "12px 14px" }}>
          <div style={{ fontSize: 11, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.4px", marginBottom: 6 }}>Exit catalyst</div>
          <div style={{ fontSize: 13, fontWeight: 500, lineHeight: 1.4 }}>{d.exit_catalyst}</div>
        </div>
        <div style={{ background: "rgba(255,255,255,0.03)", borderRadius: 8, padding: "12px 14px" }}>
          <div style={{ fontSize: 11, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.4px", marginBottom: 6 }}>Target hold</div>
          <div style={{ fontSize: 22, fontWeight: 700 }}>{d.hold_period}</div>
          <div style={{ fontSize: 11.5, color: "var(--text-3)", marginTop: 4 }}>{d.downside_protection}</div>
        </div>
      </div>

      {/* Rationale note */}
      <div style={{
        fontSize: 12.5, color: "var(--text-2)", background: "rgba(255,255,255,0.03)",
        borderRadius: 6, padding: "10px 14px", borderLeft: "3px solid var(--amber)",
      }}>{d.rationale}</div>
    </div>
  );
}

/* ====================================================================
   Expanded dimension panel — full 7-criterion deep dive
==================================================================== */
function DimDetail({ dimId, onClose, scenario }) {
  const dim = DIMS.find(d => d.id === dimId);
  const crits = CRITERIA[dimId];
  const base = DIM_STATE[dimId];
  const over = scenario && scenario.dims && scenario.dims[dimId];
  const state = over ? { ...base, ...over } : base;
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

  // Dimension-specific doc guidance
  const docGuide = {
    setup:      { title: "Setup documents", docs: "S-1, prospectus, spin announcement, index inclusions, float estimates" },
    business:   { title: "Business quality documents", docs: "10-Q, 10-K, investor presentation, SoTP, peer comp analysis" },
    capital:    { title: "Capital structure documents", docs: "10-Q (footnotes), debt agreements, pension disclosures, credit facility" },
    valuation:  { title: "Valuation documents", docs: "Investor day slides, SoTP framework, trading comps, precedent M&A" },
    incentives: { title: "Incentives documents", docs: "DEF 14A, proxy statement, equity grant schedules, insider filings" },
  }[dimId];

  const [showDocForm, setShowDocForm] = React.useState(false);
  const [docInput, setDocInput] = React.useState("");

  return (
    <div className={"m-dim-detail " + dim.cls}>
      <div className={"head " + dim.cls}>
        <span className="letter">{dim.letter}</span>
        <div>
          <div className="name">{dim.name}</div>
          <div className="keyq">{dim.key}</div>
        </div>
        <div className={"score " + scoreCls}>
          {state.score}
        </div>
        <span className="muted" style={{ fontSize: 12.5 }}>Confidence: {state.conf}</span>
        <button className="close" onClick={onClose} title={`Docs: ${docGuide.docs}`} style={{ cursor: "help" }}>?</button>
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

      {/* Setup: situation type + spin type + selling tracker */}
      {dimId === "setup" && <SetupMetaPanel />}

      <div style={{ display: "grid", gridTemplateColumns: "20px 1fr 100px 110px 120px", gap: 8 }}>
        {/* Table header */}
        <div style={{ padding: "8px 0 12px", borderBottom: "1px solid rgba(255,255,255,0.1)", fontSize: 11, fontWeight: 600, color: "#a4a4ad", textTransform: "uppercase", letterSpacing: "0.5px" }} />
        <div style={{ padding: "8px 0 12px", borderBottom: "1px solid rgba(255,255,255,0.1)", fontSize: 11, fontWeight: 600, color: "#a4a4ad", textTransform: "uppercase", letterSpacing: "0.5px" }}>Criterion</div>
        <div style={{ padding: "8px 0 12px", borderBottom: "1px solid rgba(255,255,255,0.1)", fontSize: 11, fontWeight: 600, color: "#a4a4ad", textTransform: "uppercase", letterSpacing: "0.5px", textAlign: "right" }}>Value</div>
        <div style={{ padding: "8px 0 12px", borderBottom: "1px solid rgba(255,255,255,0.1)", fontSize: 11, fontWeight: 600, color: "#a4a4ad", textTransform: "uppercase", letterSpacing: "0.5px", textAlign: "center" }}>Source</div>
        <div style={{ padding: "8px 0 12px", borderBottom: "1px solid rgba(255,255,255,0.1)", fontSize: 11, fontWeight: 600, color: "#a4a4ad", textTransform: "uppercase", letterSpacing: "0.5px", textAlign: "right" }}>Actions</div>

        {/* Table rows */}
        {crits.map((c, i) => (
          <React.Fragment key={i}>
            <span className={"status-pip " + c.status} style={{ alignSelf: "start", marginTop: 12 }} />
            <div className="name" style={{ padding: "12px 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
              {c.name}
              {c.note && <div className="sub">{c.note}</div>}
            </div>
            <div className={"value " + (c.value ? "" : "muted")} style={{ padding: "12px 0", borderBottom: "1px solid rgba(255,255,255,0.04)", textAlign: "right", fontSize: 13 }}>
              {c.value || "—"}
            </div>
            <div className={"doc-chip " + (c.docState === "pending" ? "pending" : c.docState === "missing" ? "miss" : "")} style={{ padding: "12px 0", borderBottom: "1px solid rgba(255,255,255,0.04)", textAlign: "center", display: "flex", alignItems: "center", justifyContent: "center", gap: 4 }}>
              {c.docState === "pending" ? I.clock : c.docState === "missing" ? I.alert : I.doc}
              <span>{c.doc}</span>
            </div>
            <div style={{ padding: "12px 0", borderBottom: "1px solid rgba(255,255,255,0.04)", display: "flex", gap: 4, alignItems: "center", justifyContent: "flex-end" }}>
              <button
                className="btn ghost"
                style={{ padding: "4px 6px", fontSize: 12, minWidth: 28, display: "flex", alignItems: "center", justifyContent: "center" }}
                title="Manually enter value"
              >
                ✏️
              </button>
              <button
                className="btn ghost"
                style={{ padding: "4px 6px", fontSize: 12, minWidth: 28, display: "flex", alignItems: "center", justifyContent: "center" }}
                title="Upload document"
              >
                📎
              </button>
              {c.status === "blocked" ? (
                <button className="btn" style={{ padding: "5px 10px", fontSize: 12 }}>Chase doc {I.arrowUR}</button>
              ) : c.status === "open" ? (
                <button className="btn" style={{ padding: "5px 10px", fontSize: 12 }}>Extract</button>
              ) : null}
            </div>
          </React.Fragment>
        ))}
      </div>

      {/* Valuation: peer comps table below the criteria */}
      {dimId === "valuation" && <ValuationCompsTable />}

    </div>
  );
}

/* ====================================================================
   Workspace header (composite score + state + primary actions)
==================================================================== */
function WorkspaceHeader({ onRunCommittee, committeeRunning, scenario, weightedComposite, tweaksOpen, onToggleTweaks, tweaksButtonRef }) {
  const composite  = weightedComposite || (scenario ? scenario.composite : COMPOSITE);
  const cov        = scenario ? scenario.coverage : { done: 26, total: 35, blocked: 4 };
  const stateKey   = scenario ? scenario.state : "researching";
  const stateLabel = scenario ? scenario.stateLabel : "Researching";
  const memoState  = scenario && scenario.key === "ready" ? "v3 · fresh" : "v2 · stale";
  const memoCls    = scenario && scenario.key === "ready" ? "green" : "amber";
  return (
    <div className="m-hero">
      <div className="composite-ring" style={{ "--p": (composite * 3.6) + "deg" }}>
        <span className="v">{composite}</span>
      </div>
      <div className="meta">
        <div className="label">LUMN · spinoff from CenturyTel · CIK 0000018926</div>
        <div className="name">Lumen Spinco, Inc. <span className="muted" style={{ fontWeight: 400 }}>· Telecom services</span></div>
        <div className="sub">Composite {composite} / 100 · {cov.done} of {cov.total} criteria answered{cov.blocked ? ` · ${cov.blocked} blocked` : ""}</div>
        <button ref={tweaksButtonRef} className={"btn ghost" + (tweaksOpen ? " pri" : "")} onClick={onToggleTweaks} style={{ marginTop: 8, fontSize: 12 }}>⚙ Adjust weights</button>
      </div>
      <div className="stats">
        <div className="stat">
          <div className="l">Status</div>
          <div className="v text">
            <span className={"m-state " + stateKey}><span className="dot" />{stateLabel}</span>
          </div>
        </div>
        <div className="stat">
          <div className="l">Last committee</div>
          <div className="v text">May 14 · v{scenario && scenario.key === "ready" ? 3 : 2}</div>
        </div>
        <div className="stat">
          <div className="l">Memo</div>
          <div className={"v text " + memoCls}>{memoState}</div>
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
function DocsPane({ onAdd }) {
  return (
    <div className="m-docs-pane">
      <div className="head">
        <span className="t">Documents · {DOCS.length}</span>
        <button className="btn" style={{ padding: "6px 10px", fontSize: 12.5 }} onClick={onAdd}>{I.plus} Add</button>
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

window.MeridianWorkspaceParts = { DimCard, DimDetail, WorkspaceHeader, CommitteePanel, DocsPane, ActivityFeed, QuarterlyReviewPanel, DecisionPanel };

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

window.MeridianWorkspaceParts = { DimCard, DimDetail, WorkspaceHeader, CommitteePanel, DocsPane, ActivityFeed, QuarterlyReviewPanel, DecisionPanel };

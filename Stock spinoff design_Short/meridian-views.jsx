/* global React, SpinoffUI, MeridianData, MeridianWorkspaceParts, MeridianIntake, MeridianThesis, MeridianCorpus, useTweaks, TweaksPanel, TweakSection, TweakRadio, TweakSlider */
const { I: Ic } = SpinoffUI;
const { DIMS: DIMS2, COMPOSITE: COMP } = MeridianData;

// Colored slider that matches dimension colors
function ColoredTweakSlider({ label, value, min, max, step, unit, color, onChange }) {
  const percentage = ((value - min) / (max - min)) * 100;
  const uniqueId = `slider-${label.replace(/\s+/g, '-')}`;
  return (
    <div style={{ padding: "0 14px 12px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <label style={{ fontSize: 12, fontWeight: 500, color: "#ededee" }}>{label}</label>
        <span style={{ fontSize: 11, color: "#a4a4ad" }}>{value}{unit}</span>
      </div>
      <input 
        id={uniqueId}
        type="range" 
        min={min} 
        max={max} 
        step={step} 
        value={value} 
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          width: "100%",
          height: 6,
          borderRadius: 3,
          outline: "none",
          background: `linear-gradient(to right, ${color} 0%, ${color} ${percentage}%, rgba(255,255,255,0.1) ${percentage}%, rgba(255,255,255,0.1) 100%)`,
          WebkitAppearance: "none",
          appearance: "none",
          cursor: "pointer",
        }}
      />
      <style>{`
        #${uniqueId}::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 14px;
          height: 14px;
          border-radius: 50%;
          background: ${color};
          cursor: pointer;
          box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
        #${uniqueId}::-moz-range-thumb {
          width: 14px;
          height: 14px;
          border-radius: 50%;
          background: ${color};
          cursor: pointer;
          border: none;
          box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }
      `}</style>
    </div>
  );
}

// Simple local tweaks panel (bypasses host protocol for standalone HTML)
function LocalTweaksPanel({ title, children, onClose, anchorElement }) {
  const panelRef = React.useRef(null);

  return (
    <>
      <style>{`
        .local-twk-backdrop {
          position: fixed;
          inset: 0;
          background: rgba(0, 0, 0, 0.2);
          backdrop-filter: blur(4px);
          z-index: 999;
        }
        .local-twk-panel {
          position: fixed;
          left: 50%;
          top: 50%;
          transform: translate(-50%, -50%);
          z-index: 1000;
          width: 320px;
          background: #131316;
          border: 1px solid rgba(255,255,255,0.06);
          border-radius: 12px;
          box-shadow: 0 25px 50px rgba(0,0,0,0.5);
          display: flex;
          flex-direction: column;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          color: #ededee;
          font-size: 13px;
          max-height: 80vh;
        }
        .local-twk-hd {
          padding: 16px;
          border-bottom: 1px solid rgba(255,255,255,0.06);
          display: flex;
          justify-content: space-between;
          align-items: center;
          user-select: none;
          font-weight: 600;
          font-size: 14px;
        }
        .local-twk-x {
          background: none;
          border: none;
          color: #a4a4ad;
          cursor: pointer;
          font-size: 18px;
          padding: 0;
          width: 20px;
          height: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .local-twk-x:hover {
          color: #ededee;
        }
        .local-twk-body {
          flex: 1;
          overflow-y: auto;
          padding: 12px 0;
        }
      `}</style>
      <div className="local-twk-backdrop" onClick={onClose} />
      <div ref={panelRef} className="local-twk-panel">
        <div className="local-twk-hd">
          <span>{title}</span>
          <button className="local-twk-x" onClick={onClose}>✕</button>
        </div>
        <div className="local-twk-body">
          {children}
        </div>
      </div>
    </>
  );
}
const { DimCard, DimDetail, WorkspaceHeader, CommitteePanel, DocsPane, ActivityFeed, QuarterlyReviewPanel, DecisionPanel } = MeridianWorkspaceParts;
const { Intake: IntakeV2, AddDocumentOverlay } = MeridianIntake;
const { SCENARIOS, ThesisStrengthBar, ReadyForDecisionBanner, READY: READY_THRESH } = MeridianThesis;
const { AskCorpusPanel } = MeridianCorpus;

const WORKSPACE_TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "weight_setup": 20,
  "weight_business": 20,
  "weight_capital": 20,
  "weight_valuation": 20,
  "weight_incentives": 20
}/*EDITMODE-END*/;

/* ====================================================================
   Tweak controls
==================================================================== */
function TweakSection({ title, children }) {
  return (
    <div style={{ padding: "12px 16px", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: "#a4a4ad", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 12 }}>
        {title}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {children}
      </div>
    </div>
  );
}

function TweakSlider({ label, min, max, step, value, onChange, dimColor }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 12 }}>
      <label style={{ minWidth: 80, color: "#a4a4ad" }}>{label}</label>
      <input 
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{
          flex: 1,
          height: 4,
          borderRadius: 2,
          background: "rgba(255,255,255,0.1)",
          outline: "none",
          cursor: "pointer",
          appearance: "none",
          WebkitAppearance: "none",
          accentColor: dimColor || "#4a90e2",
        }}
      />
      <span style={{ minWidth: 30, textAlign: "right", color: "#ededee", fontWeight: 500 }}>
        {Math.round(value)}%
      </span>
    </div>
  );
}

/* ====================================================================
   The Workspace view — the heart of Meridian-SS
==================================================================== */
function Workspace({ ticker, goNewIntake, openQuarterly, isBlank = false, goBack }) {
  const [focused, setFocused] = React.useState(null);
  const [activeTab, setActiveTab] = React.useState("dims"); // dims, corpus, committee, journal
  const [committee, setCommittee] = React.useState(false);
  const [coachOpen, setCoachOpen] = React.useState(true);
  const [quarterly, setQuarterly] = React.useState(!!openQuarterly);
  const [quarterlyDismissed, setQuarterlyDismissed] = React.useState(false);
  const [addDocOpen, setAddDocOpen] = React.useState(false);
  const [askOpen, setAskOpen] = React.useState(!isBlank);
  const [tweaksOpen, setTweaksOpen] = React.useState(false);
  const [workspaceTicker, setWorkspaceTicker] = React.useState(ticker || "");
  const [parentCompany, setParentCompany] = React.useState("");
  const tweaksButtonRef = React.useRef(null);

  const [t, setTweak] = useTweaks(WORKSPACE_TWEAK_DEFAULTS);
  const scenario = isBlank ? { ...SCENARIOS.researching, composite: 0 } : SCENARIOS.researching;
  const isReady = false;
  const isStrengthening = false;

  React.useEffect(() => { if (openQuarterly) setQuarterly(true); }, [openQuarterly]);

  const weights = {
    setup: t.weight_setup,
    business: t.weight_business,
    capital: t.weight_capital,
    valuation: t.weight_valuation,
    incentives: t.weight_incentives,
  };
  const totalWeight = weights.setup + weights.business + weights.capital + weights.valuation + weights.incentives;
  const weightedComposite = totalWeight > 0
    ? Math.round((
        (DIMS2[0].id === "setup" ? DIM_STATE.setup.score : 0) * weights.setup +
        (DIMS2[1].id === "business" ? DIM_STATE.business.score : 0) * weights.business +
        (DIMS2[2].id === "capital" ? DIM_STATE.capital.score : 0) * weights.capital +
        (DIMS2[3].id === "valuation" ? DIM_STATE.valuation.score : 0) * weights.valuation +
        (DIMS2[4].id === "incentives" ? DIM_STATE.incentives.score : 0) * weights.incentives
      ) / totalWeight)
    : 0;

  return (
    <div>
      {/* Header */}
      {isBlank ? (
        <div style={{ paddingBottom: 24, borderBottom: "1px solid rgba(255,255,255,0.06)", marginBottom: 24 }}>
          <div style={{ display: "flex", gap: 20, alignItems: "flex-end" }}>
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 12, color: "#a4a4ad", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8 }}>Ticker</div>
              <input 
                type="text" 
                placeholder="e.g. LUMN, VNTR, STHO" 
                value={workspaceTicker}
                onChange={(e) => setWorkspaceTicker(e.target.value.toUpperCase())}
                style={{ 
                  fontSize: 24, 
                  fontWeight: 600, 
                  background: "none", 
                  border: "none", 
                  color: "#ededee", 
                  outline: "none",
                  padding: 0,
                  marginBottom: 8,
                  width: "100%"
                }} 
              />
              <input 
                type="text" 
                placeholder="Parent company (optional)" 
                value={parentCompany}
                onChange={(e) => setParentCompany(e.target.value)}
                style={{ 
                  fontSize: 13, 
                  color: "#a4a4ad", 
                  background: "none", 
                  border: "none", 
                  outline: "none",
                  padding: 0,
                  width: "100%"
                }} 
              />
            </div>
            <div>
              <button className="btn pri" onClick={() => goBack && goBack()}>Back to portfolio</button>
            </div>
          </div>
        </div>
      ) : (
        <WorkspaceHeader scenario={scenario} weightedComposite={weightedComposite} 
          tweaksOpen={tweaksOpen} onToggleTweaks={() => setTweaksOpen(v => !v)} tweaksButtonRef={tweaksButtonRef} />
      )}

      {/* Tabs */}
      <div style={{ display: "flex", gap: 0, borderBottom: "1px solid rgba(255,255,255,0.06)", marginBottom: 24, fontSize: 13 }}>
        {[
          { k: "dims", label: "📊 Dimensions", icon: "📊" },
          { k: "corpus", label: "💬 Ask corpus", icon: "💬" },
          { k: "committee", label: "🤖 Committee", icon: "🤖" },
          { k: "journal", label: "📖 Journal", icon: "📖" },
        ].map(t => (
          <button 
            key={t.k}
            onClick={() => setActiveTab(t.k)}
            style={{
              padding: "12px 16px",
              border: "none",
              borderBottom: activeTab === t.k ? "2px solid #4a90e2" : "2px solid transparent",
              background: "none",
              color: activeTab === t.k ? "#ededee" : "#a4a4ad",
              cursor: "pointer",
              fontSize: 13,
              fontWeight: activeTab === t.k ? 500 : 400,
              transition: "all 120ms"
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Content based on active tab */}
      {activeTab === "dims" && (
        <div>
          <div className="m-dim-grid">
            {DIMS2.map(d => (
              <DimCard key={d.id} dim={d} scenario={scenario}
                focused={focused === d.id}
                onFocus={(id) => setFocused(focused === id ? null : id)} />
            ))}
          </div>

          {focused && <DimDetail dimId={focused} scenario={scenario} onClose={() => setFocused(null)} />}

          <div className="m-actions" style={{ marginTop: 20, marginBottom: 4, justifyContent: "flex-end" }}>
            <button className="btn" onClick={() => setAddDocOpen(true)}>{Ic.plus} Add document</button>
            <button className="btn">{Ic.copy2} Memo</button>
          </div>

          {addDocOpen && <AddDocumentOverlay onClose={() => setAddDocOpen(false)} />}

          <DecisionPanel />
        </div>
      )}

      {activeTab === "corpus" && (
        <div style={{ marginBottom: 28 }}>
          <AskCorpusPanel onClose={() => {}} />
        </div>
      )}

      {activeTab === "committee" && (
        <div style={{ marginBottom: 28 }}>
          <div className="m-actions" style={{ marginBottom: 20 }}>
            <button className="btn pri" onClick={() => setCommittee(true)}>Run committee {Ic.arrowR}</button>
          </div>
          {committee && <CommitteePanel onClose={() => setCommittee(false)} />}
        </div>
      )}

      {activeTab === "journal" && (
        <div style={{ marginBottom: 28 }}>
          <DecisionJournal ready={isReady} ticker={workspaceTicker} />
        </div>
      )}

      {tweaksOpen && (
        <LocalTweaksPanel title="Adjust weights" onClose={() => setTweaksOpen(false)} anchorElement={tweaksButtonRef.current}>
          <TweakSection title="Dimension weights">
            <TweakSlider label="Setup" min={0} max={100} step={1} value={t.weight_setup} onChange={(v) => setTweak("weight_setup", v)} dimColor="oklch(0.72 0.15 295)" />
            <TweakSlider label="Business" min={0} max={100} step={1} value={t.weight_business} onChange={(v) => setTweak("weight_business", v)} dimColor="oklch(0.72 0.13 240)" />
            <TweakSlider label="Capital" min={0} max={100} step={1} value={t.weight_capital} onChange={(v) => setTweak("weight_capital", v)} dimColor="oklch(0.78 0.15 75)" />
            <TweakSlider label="Valuation" min={0} max={100} step={1} value={t.weight_valuation} onChange={(v) => setTweak("weight_valuation", v)} dimColor="oklch(0.78 0.16 148)" />
            <TweakSlider label="Incentives" min={0} max={100} step={1} value={t.weight_incentives} onChange={(v) => setTweak("weight_incentives", v)} dimColor="oklch(0.72 0.16 350)" />
          </TweakSection>
        </LocalTweaksPanel>
      )}
    </div>
  );
}

/* ====================================================================
   Decision journal — full page view showing chronological list
==================================================================== */
function DecisionJournal({ ready, ticker }) {
  const [selectedEntry, setSelectedEntry] = React.useState(ticker ? `${ticker}-v1` : "LUMN-v3");
  const [conv, setConv] = React.useState(8);

  const allEntries = [
    { id: "LUMN-v3", ticker: "LUMN", company: "Lumen Spinco", date: "Today", conviction: 7, composite: 65, status: "Ready" },
    { id: "LUMN-v2", ticker: "LUMN", company: "Lumen Spinco", date: "May 20", conviction: 6, composite: 62, status: "Researching" },
    { id: "LUMN-v1", ticker: "LUMN", company: "Lumen Spinco", date: "May 15", conviction: 5, composite: 58, status: "Researching" },
    { id: "VNTR-v1", ticker: "VNTR", company: "Vontier", date: "Yesterday", conviction: 6, composite: 58, status: "Researching" },
    { id: "STHO-v3", ticker: "STHO", company: "Stericycle Hldg", date: "May 18", conviction: 8, composite: 82, status: "Ready" },
    { id: "APP-v1", ticker: "APP", company: "Applicant Pro", date: "May 10", conviction: 5, composite: 55, status: "Watching" },
    { id: "SMPL-v2", ticker: "SMPL", company: "Sample Co", date: "May 5", conviction: 6, composite: 68, status: "Watching" },
  ];

  // Filter by ticker if provided
  const entries = ticker ? allEntries.filter(e => e.ticker === ticker) : allEntries;
  const current = entries.find(e => e.id === selectedEntry) || entries[0];

  return (
    <div style={{ display: "grid", gridTemplateColumns: ticker ? "1fr" : "280px 1fr", gap: 24, height: "100%" }}>
      {/* Entry List — only show if multiple tickers or in Decision Journal view */}
      {!ticker && (
        <div style={{ background: "var(--bg-elev)", border: "1px solid var(--border)", borderRadius: 12, padding: 16, overflow: "auto", display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            📖 Entries
            <span style={{ fontSize: 11, color: "var(--text-3)" }}>{entries.length}</span>
          </div>
          {entries.map(e => (
            <div 
              key={e.id}
              onClick={() => setSelectedEntry(e.id)}
              style={{
                padding: 10,
                borderRadius: 8,
                background: selectedEntry === e.id ? "rgba(74, 144, 226, 0.15)" : "rgba(255,255,255,0.02)",
                border: selectedEntry === e.id ? "1px solid rgba(74, 144, 226, 0.3)" : "1px solid transparent",
                cursor: "pointer",
                transition: "all 120ms"
              }}
            >
              <div style={{ fontSize: 11, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 2 }}>{e.date}</div>
              <div style={{ fontSize: 13, fontWeight: 600, color: "var(--text)" }}>{e.ticker}</div>
              <div style={{ fontSize: 11, color: "var(--text-2)", marginTop: 2 }}>{e.company}</div>
            </div>
          ))}
        </div>
      )}

      {/* Entry Detail */}
      <div style={{ background: "var(--bg-elev)", border: "1px solid var(--border)", borderRadius: 12, padding: 24, overflow: "auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, marginBottom: 24, paddingBottom: 24, borderBottom: "1px solid var(--border)" }}>
          <div>
            <div style={{ fontSize: 24, fontWeight: 700, color: "var(--text)", marginBottom: 4 }}>{current.ticker}</div>
            <div style={{ fontSize: 13, color: "var(--text-2)" }}>{current.company} · {current.date}</div>
          </div>
          <div style={{
            fontSize: 11,
            padding: "6px 10px",
            borderRadius: 6,
            background: current.status === "Ready" ? "rgba(16, 185, 129, 0.15)" : current.status === "Researching" ? "rgba(74, 144, 226, 0.15)" : "rgba(255, 159, 64, 0.15)",
            color: current.status === "Ready" ? "var(--green)" : current.status === "Researching" ? "var(--blue)" : "var(--amber)",
          }}>
            {current.status === "Ready" ? "✓" : "●"} {current.status}
          </div>
        </div>

        {/* Metrics */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, marginBottom: 24 }}>
          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", borderRadius: 8, padding: 12 }}>
            <div style={{ fontSize: 10, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 6 }}>Conviction</div>
            <div style={{ fontSize: 20, fontWeight: 700, color: "var(--amber)" }}>{current.conviction} / 10</div>
          </div>
          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", borderRadius: 8, padding: 12 }}>
            <div style={{ fontSize: 10, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 6 }}>Composite</div>
            <div style={{ fontSize: 20, fontWeight: 700, color: current.composite >= 80 ? "var(--green)" : current.composite >= 60 ? "var(--amber)" : "var(--red)" }}>{current.composite}</div>
          </div>
          <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)", borderRadius: 8, padding: 12 }}>
            <div style={{ fontSize: 10, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 6 }}>Coverage</div>
            <div style={{ fontSize: 20, fontWeight: 700, color: "var(--text)" }}>26 / 35</div>
          </div>
        </div>

        {/* Details */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <div>
            <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8 }}>Primary driver</div>
            <div style={{ fontSize: 13, color: "var(--text-2)", lineHeight: 1.6 }}>
              Forced-selling spinoff with above-peer business quality at attractive multiples. Capital structure de-risked post-pension disclosure.
            </div>
          </div>

          <div>
            <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8 }}>Key acknowledged risks</div>
            <div style={{ background: "rgba(74, 144, 226, 0.1)", borderLeft: "3px solid var(--blue)", padding: 12, borderRadius: 4, fontSize: 12, color: "var(--text-2)", lineHeight: 1.6 }}>
              Index-fund forced selling could overshoot; sized at 7% to absorb a 15% drawdown. Pension transfer still materializing — committee capped Capital score at 60 pending May 23 footnote.
            </div>
          </div>

          <div>
            <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-3)", textTransform: "uppercase", letterSpacing: "0.5px", marginBottom: 8 }}>Monitoring checkpoints</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div style={{ display: "flex", gap: 8, padding: 8, background: "rgba(255,255,255,0.02)", borderRadius: 6 }}>
                <span style={{ minWidth: 16 }}>📋</span>
                <div style={{ fontSize: 12, color: "var(--text-2)" }}><strong>Q2 10-Q</strong> · Pension funding plan disclosure (due May 23)</div>
              </div>
              <div style={{ display: "flex", gap: 8, padding: 8, background: "rgba(255,255,255,0.02)", borderRadius: 6 }}>
                <span style={{ minWidth: 16 }}>🎤</span>
                <div style={{ fontSize: 12, color: "var(--text-2)" }}><strong>Investor Day</strong> · SoTP framework from mgmt (due Q1)</div>
              </div>
              <div style={{ display: "flex", gap: 8, padding: 8, background: "rgba(255,255,255,0.02)", borderRadius: 6 }}>
                <span style={{ minWidth: 16 }}>📊</span>
                <div style={{ fontSize: 12, color: "var(--text-2)" }}><strong>Margin Trend</strong> · Watch for &lt;-150 bps YoY deterioration</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ====================================================================
   Portfolio view — list of workspaces
==================================================================== */
function Portfolio({ onOpen, onNew }) {
  const [tab, setTab] = React.useState("active");
  
  const positions = [
    { ticker: "LUMN",  name: "Lumen Spinco",  composite: 65, state: "researching", states: { S: 88, B: 82, C: 45, V: 60, I: 48 }, played: { S: true }, blocked: 4, last: "Today", status: "active" },
    { ticker: "VNTR",  name: "Vontier",       composite: 58, state: "invested", states: { S: "Played", B: 62, C: 45, V: 60, I: 48 }, played: { S: true }, blocked: 1, last: "May 9", status: "active" },
    { ticker: "STHO",  name: "Stericycle Hldg",composite: 82, state: "invested", states: { S: "Played", B: 86, C: 78, V: 82, I: 80 }, played: { S: true }, blocked: 0, last: "Q4 '25", status: "active" },
    { ticker: "LTRPA", name: "Liberty Tripadvisor", composite: 74, state: "memo-ready", states: { S: 80, B: 78, C: 62, V: 70, I: 80 }, played: {}, blocked: 0, last: "May 18", status: "closed" },
    { ticker: "APP",   name: "Applicant Pro",  composite: 55, state: "watching", states: { S: 72, B: 58, C: 45, V: 60, I: 48 }, played: {}, blocked: 2, last: "May 10", status: "watching" },
    { ticker: "SMPL",  name: "Sample Co",     composite: 68, state: "watching", states: { S: 75, B: 70, C: 65, V: 68, I: 65 }, played: {}, blocked: 1, last: "May 5", status: "watching" },
  ];

  const filteredPositions = positions.filter(p => p.status === tab);

  return (
    <div>
      <div className="m-page-head">
        <div>
          <h1 className="h1">Portfolio Hub</h1>
          <div className="sub">{positions.length} total analyses across Active, Closed, and Watching</div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 0, borderBottom: "1px solid var(--border)", marginBottom: 24, fontSize: 13 }}>
        {[
          { k: "active", label: "📌 Active", count: positions.filter(p => p.status === "active").length },
          { k: "closed", label: "✓ Closed", count: positions.filter(p => p.status === "closed").length },
          { k: "watching", label: "👁️ Watching", count: positions.filter(p => p.status === "watching").length },
        ].map(t => (
          <button 
            key={t.k}
            onClick={() => setTab(t.k)}
            style={{
              padding: "12px 16px",
              border: "none",
              borderBottom: tab === t.k ? "2px solid var(--blue)" : "2px solid transparent",
              background: "none",
              color: tab === t.k ? "var(--text)" : "var(--text-2)",
              cursor: "pointer",
              fontSize: 13,
              fontWeight: tab === t.k ? 500 : 400,
              transition: "all 120ms"
            }}
          >
            {t.label} ({t.count})
          </button>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: 20 }}>
        {filteredPositions.map(p => (
          <div 
            key={p.ticker} 
            style={{
              background: "var(--bg-elev)",
              border: "1px solid var(--border)",
              borderRadius: 12,
              padding: 20,
              transition: "all 120ms",
              display: "flex",
              flexDirection: "column",
              gap: 12
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "var(--bg-elev-2)";
              e.currentTarget.style.borderColor = "rgba(255,255,255,0.1)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "var(--bg-elev)";
              e.currentTarget.style.borderColor = "var(--border)";
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
              <div>
                <div style={{ fontSize: 12, color: "var(--text-2)" }}>{p.ticker}</div>
                <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text)" }}>{p.name}</div>
              </div>
              <div style={{
                fontSize: 11,
                padding: "4px 8px",
                borderRadius: 4,
                background: "rgba(255,255,255,0.05)",
                color: "var(--text-2)",
                whiteSpace: "nowrap"
              }}>
                {tab === "active" ? "● Active" : tab === "closed" ? "✓ Closed" : "👁️ Watching"}
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, fontSize: 12 }}>
              <div>
                <div style={{ color: "var(--text-3)", textTransform: "uppercase", fontSize: 10, letterSpacing: "0.5px", marginBottom: 6 }}>Composite</div>
                <div style={{ fontSize: 24, fontWeight: 700, color: p.composite >= 80 ? "var(--green)" : p.composite >= 60 ? "var(--amber)" : "var(--red)" }}>
                  {p.composite}
                </div>
              </div>
              <div>
                <div style={{ color: "var(--text-3)", textTransform: "uppercase", fontSize: 10, letterSpacing: "0.5px", marginBottom: 6 }}>Status</div>
                <div style={{ color: "var(--text)" }}>
                  {p.state === "researching" ? "Researching" : p.state === "memo-ready" ? "Memo ready" : "Invested"}
                </div>
              </div>
            </div>

            <div>
              <div style={{ color: "var(--text-3)", textTransform: "uppercase", fontSize: 10, letterSpacing: "0.5px", marginBottom: 8 }}>Dimensions</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 8 }}>
                {DIMS2.map(d => {
                  const v = p.states[d.letter];
                  const isText = typeof v !== "number";
                  return (
                    <div 
                      key={d.id} 
                      style={{
                        background: `var(--d-${d.id}-soft)`,
                        border: "1px solid rgba(255,255,255,0.06)",
                        borderRadius: 8,
                        padding: 10,
                        textAlign: "center"
                      }}
                    >
                      <div style={{ fontSize: 12, fontWeight: 600, color: `var(--d-${d.id})`, marginBottom: 4 }}>{d.letter}</div>
                      <div style={{ 
                        fontSize: 18, 
                        fontWeight: 600, 
                        color: isText ? "var(--text)" : (v >= 80 ? "var(--green)" : v >= 60 ? "var(--amber)" : "var(--red)")
                      }}>
                        {isText ? v : v}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div style={{ borderTop: "1px solid rgba(255,255,255,0.04)", paddingTop: 12 }}>
              <button 
                className="btn pri"
                onClick={() => onOpen(p.ticker)}
                style={{ width: "100%", justifyContent: "center" }}
              >
                Open Analysis
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

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
   New Analysis intake — multi-step (v2)
   Implementation lives in meridian-intake.jsx; re-exported here so
   meridian-app.jsx can keep destructuring { Intake } from MeridianViews.
==================================================================== */
const Intake = IntakeV2;

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


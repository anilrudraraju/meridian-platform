/* global React, ReactDOM, SpinoffUI, MeridianViews */
const { I: NavI } = SpinoffUI;
const { Workspace, Portfolio, Updates, Intake, Dev } = MeridianViews;

function Sidebar({ view, setView, onNew }) {
  const items = [
    { k: "portfolio", label: "Portfolio Hub",  ico: NavI.grid,  count: 23 },
    { k: "journal",   label: "Decision Journal", ico: NavI.bookmark, count: 23 },
  ];
  return (
    <aside className="m-side">
      <div className="m-brand">
        <span className="glyph" />
        <div>
          <div className="name">Meridian-SS</div>
          <div className="ver">v0.1 · special situations</div>
        </div>
      </div>

      <button className="btn pri" style={{ justifyContent: "center", marginBottom: 12 }} onClick={onNew}>
        {NavI.plus} New analysis
      </button>

      <nav className="m-nav">
        {items.map(it => (
          <div key={it.k}
            className={"item " + (view === it.k ? "active" : "")}
            onClick={() => setView(it.k)}>
            <span className="ico">{it.ico}</span>
            <span>{it.label}</span>
            {it.badge ? <span className="badge">{it.badge}</span>
                      : it.count != null ? <span className="count">{it.count}</span> : null}
          </div>
        ))}
      </nav>

      <div className="m-side-foot">
        <div className="m-key-status">
          <span className="dot" />
          <span style={{ flex: 1 }}>API keys connected</span>
          <span className="muted" style={{ fontSize: 11.5 }}>2/2</span>
        </div>
        <div className="muted" style={{ fontSize: 11, textAlign: "center", marginTop: 4 }}>
          meridian.db · 4 tickers · 27 docs
        </div>
      </div>
    </aside>
  );
}

function App() {
  const [view, setView] = React.useState("portfolio");
  const [isNewAnalysis, setIsNewAnalysis] = React.useState(false);
  const [openQuarterly, setOpenQuarterly] = React.useState(false);
  const [activeWorkspaceTicker, setActiveWorkspaceTicker] = React.useState(null);

  const navigate = (v, opts = {}) => {
    setView(v);
    setIsNewAnalysis(false);
    if (opts.quarterly) setOpenQuarterly(true);
    else setOpenQuarterly(false);
  };

  return (
    <div className="m-app">
      <Sidebar view={view} setView={navigate} onNew={() => { setIsNewAnalysis(true); setView("workspace"); setActiveWorkspaceTicker(null); }} />
      <main className="m-main" data-screen-label={view}>
        {view === "portfolio" && <Portfolio onOpen={(ticker) => { setActiveWorkspaceTicker(ticker); setView("workspace"); }} onNew={() => setIsNewAnalysis(true)} />}
        {view === "journal"   && <DecisionJournal ready={false} />}
        {view === "workspace" && <Workspace ticker={activeWorkspaceTicker} goNewIntake={() => { setIsNewAnalysis(true); setActiveWorkspaceTicker(null); }} openQuarterly={openQuarterly} isBlank={isNewAnalysis} goBack={() => { setView("portfolio"); setIsNewAnalysis(false); }} />}
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

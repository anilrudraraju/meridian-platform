/* global React, ReactDOM, SpinoffUI, MeridianViews */
const { I: NavI } = SpinoffUI;
const { Workspace, Portfolio, Updates, Intake, Dev } = MeridianViews;

function Sidebar({ view, setView, onNew }) {
  const items = [
    { k: "workspace", label: "Workspace",  ico: NavI.eye,      count: "LUMN" },
    { k: "portfolio", label: "Portfolio",  ico: NavI.grid,     count: 7 },
    { k: "updates",   label: "Updates",    ico: NavI.inbox,    badge: 4 },
    { k: "dev",       label: "Dev",        ico: NavI.folder },
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
  const [view, setView] = React.useState("workspace");
  const [intakeOpen, setIntakeOpen] = React.useState(false);
  const [openQuarterly, setOpenQuarterly] = React.useState(false);

  const navigate = (v, opts = {}) => {
    setView(v);
    if (opts.quarterly) setOpenQuarterly(true);
    else setOpenQuarterly(false);
  };

  if (intakeOpen) {
    return (
      <Intake onCancel={() => setIntakeOpen(false)}
              onCreate={() => { setIntakeOpen(false); setView("workspace"); }} />
    );
  }

  return (
    <div className="m-app">
      <Sidebar view={view} setView={setView} onNew={() => setIntakeOpen(true)} />
      <main className="m-main" data-screen-label={view}>
        {view === "workspace" && <Workspace goNewIntake={() => setIntakeOpen(true)} openQuarterly={openQuarterly} />}
        {view === "portfolio" && <Portfolio onOpen={() => navigate("workspace")} onNew={() => setIntakeOpen(true)} />}
        {view === "updates"   && <Updates onOpen={(_, opts) => navigate("workspace", opts)} />}
        {view === "dev"       && <Dev />}
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

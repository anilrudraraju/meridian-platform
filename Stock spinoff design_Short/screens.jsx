/* global React, SpinoffUI */
const { I, Dims } = SpinoffUI;

/* ---------- Dashboard ---------- */
function Dashboard({ openCompany, goToUpdates }) {
  const portfolio = [
    { ticker: "LUMN",  type: "Spinoff",  score: 78, dims: ["g","g","a","g","g"], status: "intact", q: "Q1 '26" },
    { ticker: "VNTR",  type: "Spinoff",  score: 58, dims: ["x","a","r","a","r"], status: "review", q: "Q1 '26" },
    { ticker: "STHO",  type: "Spinoff",  score: 82, dims: ["g","g","g","g","g"], status: "intact", q: "Q4 '25" },
    { ticker: "LTRPA", type: "Tracking", score: 74, dims: ["g","g","a","g","g"], status: "intact", q: "Q1 '26" },
  ];
  return (
    <div>
      <div className="kpi-grid">
        <div className="kpi"><div className="label">Invested</div><div className="val">7</div></div>
        <div className="kpi"><div className="label">Watchlist</div><div className="val">14</div></div>
        <div className="kpi"><div className="label">Pending review</div><div className="val amber">3</div></div>
        <div className="kpi"><div className="label">Avg score</div><div className="val">71</div></div>
      </div>

      <h2 className="section-h" style={{ marginTop: 28 }}>Portfolio</h2>
      <div className="card">
        <table className="tbl">
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Type</th>
              <th>Score</th>
              <th>Dimensions (S/B/C/V/I)</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {portfolio.map(p => (
              <tr key={p.ticker} onClick={() => openCompany(p.ticker)}>
                <td className="ticker">{p.ticker}</td>
                <td className="muted">{p.type}</td>
                <td className="score">{p.score}</td>
                <td><Dims vals={p.dims} /></td>
                <td>
                  {p.status === "intact"
                    ? <span className="pill green">Intact</span>
                    : <span className="pill amber">Review</span>}
                </td>
                <td className="muted" style={{ textAlign: "right" }}>{p.q}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="row" style={{ justifyContent: "space-between", margin: "32px 0 12px" }}>
        <h2 className="section-h" style={{ margin: 0 }}>Pending review</h2>
        <div className="muted" style={{ fontSize: 13.5 }}>3 updates · click to go to the inbox</div>
      </div>
      <div className="pending-banner" onClick={goToUpdates}>
        <div className="ico-wrap">{I.doc}</div>
        <div>
          <div className="title">VNTR Q1 2026 quarterly update ready</div>
          <div className="sub">Committee flags 2 missed promises. + 2 more pending updates.</div>
        </div>
        <div className="muted">{I.arrowR}</div>
      </div>

      <div className="muted" style={{ fontSize: 12.5, marginTop: 18 }}>
        Dimensions key: <b style={{ color: "var(--text-2)" }}>S</b> Setup · <b style={{ color: "var(--text-2)" }}>B</b> Business · <b style={{ color: "var(--text-2)" }}>C</b> Capital · <b style={{ color: "var(--text-2)" }}>V</b> Valuation · <b style={{ color: "var(--text-2)" }}>I</b> Incentives. Click any row to open the company detail.
      </div>
    </div>
  );
}

/* ---------- Updates ---------- */
function Updates({ openCompany }) {
  const [filter, setFilter] = React.useState("All");
  const tabs = [
    { k: "All",          n: 3 },
    { k: "Quarterly",    n: 1 },
    { k: "8-K",          n: 1 },
    { k: "Pending docs", n: 1 },
  ];
  return (
    <div>
      <div className="upd-tabs">
        {tabs.map(t => (
          <button key={t.k}
            className={"upd-tab " + (filter === t.k ? "active" : "")}
            onClick={() => setFilter(t.k)}>
            {t.k} · {t.n}
          </button>
        ))}
      </div>

      <div className="upd-card" onClick={() => openCompany("VNTR")} style={{ cursor: "pointer" }}>
        <div className="head">
          <div className="ico-wrap">{I.doc}</div>
          <div>
            <div className="title">VNTR Q1 2026 quarterly update</div>
            <div className="sub">10-Q filed May 9 · click to open VNTR detail</div>
          </div>
          <span className="pill amber">Review</span>
        </div>
        <div className="panel">
          <div className="h">5-dimension score change vs. v2</div>
          <div className="score-grid">
            <div className="score-tile">
              <div className="label">Setup</div>
              <div className="val text">Played</div>
            </div>
            <div className="score-tile">
              <div className="label">Business</div>
              <div className="val">62<span className="delta red">−8</span></div>
            </div>
            <div className="score-tile">
              <div className="label">Capital</div>
              <div className="val">45<span className="delta red">−15</span></div>
            </div>
            <div className="score-tile">
              <div className="label">Valuation</div>
              <div className="val">60<span className="delta amber">−5</span></div>
            </div>
            <div className="score-tile">
              <div className="label">Incentives</div>
              <div className="val">48<span className="delta red">−12</span></div>
            </div>
          </div>
          <div className="muted" style={{ fontSize: 13.5, marginTop: 14 }}>
            Memo v3 ready · click anywhere on this card to open VNTR detail
          </div>
        </div>
      </div>

      <div className="upd-card">
        <div className="head bell">
          <div className="ico-wrap">{I.bell}</div>
          <div>
            <div className="title">LUMN · 8-K filed May 18, 2026</div>
            <div className="sub">Material agreement amendment. Add the 8-K and any related news, then re-run the committee.</div>
          </div>
          <div />
        </div>
        <div className="actions">
          <button className="btn">Dismiss</button>
          <button className="btn">Add documents and re-run {I.arrowUR}</button>
        </div>
      </div>

      <div className="upd-card">
        <div className="head clock">
          <div className="ico-wrap">{I.clock}</div>
          <div>
            <div className="title">LUMN · pension footnote expected today</div>
            <div className="sub">You flagged this missing on Apr 22. Has it been disclosed?</div>
          </div>
          <div />
        </div>
        <div className="actions">
          <button className="btn">Still missing</button>
          <button className="btn">Check EDGAR {I.arrowUR}</button>
          <button className="btn">Upload now {I.arrowUR}</button>
        </div>
      </div>
    </div>
  );
}

/* ---------- Company detail ---------- */
function CompanyDetail() {
  const [section, setSection] = React.useState("Overview");
  const nav = [
    { k: "Overview",  ico: I.eye, count: null },
    { k: "Memos",     ico: I.copy2, count: 3 },
    { k: "Scorecard", ico: I.chart, count: null },
    { k: "Metrics",   ico: I.line,  count: null },
    { k: "Promises",  ico: I.check2, count: 5 },
    { k: "Documents", ico: I.folder, count: 18 },
    { k: "Notes",     ico: I.bookmark, count: 12 },
    { k: "Q&A",       ico: I.chat, count: 27 },
  ];
  return (
    <div>
      <div className="cd-head">
        <div>
          <div className="h1">
            <span className="ticker">VNTR</span>
            <span className="muted">·</span>
            <span>Vontier</span>
            <span className="pill amber">Invested · review</span>
          </div>
          <div className="sub">Spinoff from Fortive (FTV) · Composite score 58 / 100</div>
        </div>
        <div className="actions">
          <button className="btn">{I.refresh} Re-run</button>
          <button className="btn">{I.download} Export</button>
        </div>
      </div>

      <div className="cd-grid">
        <div className="cd-side">
          {nav.map(n => (
            <div key={n.k} className={"item " + (section === n.k ? "active" : "")} onClick={() => setSection(n.k)}>
              <span className="ico">{n.ico}</span>
              <span>{n.k}</span>
              {n.count != null ? <span className="count">{n.count}</span> : null}
            </div>
          ))}
        </div>

        <div>
          <h3 className="section-h">Scorecard summary</h3>
          <div className="score-grid">
            <div className="score-tile">
              <div className="label">Setup</div>
              <div className="val text">Played</div>
              <div className="bar"><span className="g" style={{ width: "100%", background: "rgba(255,255,255,0.18)" }} /></div>
            </div>
            <div className="score-tile">
              <div className="label">Business</div>
              <div className="val">62</div>
              <div className="bar"><span className="a" style={{ width: "62%" }} /></div>
            </div>
            <div className="score-tile">
              <div className="label">Capital</div>
              <div className="val">45</div>
              <div className="bar"><span className="r" style={{ width: "45%" }} /></div>
            </div>
            <div className="score-tile">
              <div className="label">Valuation</div>
              <div className="val">60</div>
              <div className="bar"><span className="a" style={{ width: "60%" }} /></div>
            </div>
            <div className="score-tile">
              <div className="label">Incentives</div>
              <div className="val">48</div>
              <div className="bar"><span className="r" style={{ width: "48%" }} /></div>
            </div>
          </div>

          <h3 className="section-h" style={{ marginTop: 24 }}>Business quality detail</h3>
          <div className="bq">
            <div className="col"><div className="l">ROIC</div><div className="v">15.8%</div></div>
            <div className="col"><div className="l">Growth</div><div className="v">2.1%</div></div>
            <div className="col"><div className="l">FCF conv.</div><div className="v">78%</div></div>
            <div className="col"><div className="l">Margin</div><div className="v text">Deteriorating</div></div>
            <div className="col"><div className="l">Durability</div><div className="v">3 / 5</div></div>
          </div>
        </div>
      </div>
    </div>
  );
}

window.Screens = { Dashboard, Updates, CompanyDetail };

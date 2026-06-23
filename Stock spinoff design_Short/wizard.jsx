/* global React, SpinoffUI */
const { I } = SpinoffUI;

/* ---------- New analysis wizard ---------- */
function Wizard({ goToCompany }) {
  // step state: 1..6, plus highest "completed" boundary
  const [step, setStep] = React.useState(1);
  const [maxDone, setMaxDone] = React.useState(0);
  const advance = (next) => {
    setMaxDone(d => Math.max(d, step));
    setStep(next);
  };

  const steps = [
    { n: 1, title: "Intake",     sub: "30 sec" },
    { n: 2, title: "Documents",  sub: "2 min" },
    { n: 3, title: "Ingest",     sub: "3 min" },
    { n: 4, title: "Explore",    sub: "optional" },
    { n: 5, title: "Committee",  sub: "4 min" },
    { n: 6, title: "Memo",       sub: "review" },
  ];

  return (
    <div>
      <h1 style={{ fontSize: 22, fontWeight: 500, letterSpacing: "-0.02em", margin: "0 0 6px" }}>
        LUMN spinoff from CenturyTel
      </h1>
      <div className="row" style={{ color: "var(--amber)", fontSize: 14, marginBottom: 22 }}>
        <span style={{ width: 16, height: 16, borderRadius: 999, border: "1.5px solid currentColor", display: "inline-flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 600 }}>i</span>
        <span>Analysis is marked complete only after all 6 steps and your final invest / watch / reject decision.</span>
      </div>

      <div className="wiz-grid">
        <div className="wiz-side">
          {steps.map(s => {
            const isCurrent = s.n === step;
            const isDone = s.n <= maxDone && !isCurrent;
            return (
              <div key={s.n}
                className={"wiz-step " + (isCurrent ? "current " : "") + (isDone ? "done" : "")}
                onClick={() => setStep(s.n)}>
                <div className="dot">{isDone ? <span style={{ fontSize: 14 }}>✓</span> : s.n}</div>
                <div>
                  <div className="title">{s.title}</div>
                  <div className="sub">{s.sub}</div>
                </div>
              </div>
            );
          })}
        </div>

        <div className="wiz-panel">
          {step === 1 && <Step1 onNext={() => advance(2)} />}
          {step === 2 && <Step2 onNext={() => advance(3)} />}
          {step === 3 && <Step3 onNext={() => advance(4)} />}
          {step === 4 && <Step4 onNext={() => advance(5)} />}
          {step === 5 && <Step5 onNext={() => advance(6)} />}
          {step === 6 && <Step6 onDone={goToCompany} />}
        </div>
      </div>
    </div>
  );
}

/* ---------- Step 1: Intake ---------- */
function Step1({ onNext }) {
  const [ticker, setTicker] = React.useState("LUMN");
  const [parent, setParent] = React.useState("CTL");
  const [situation, setSituation] = React.useState("Spinoff");
  const [url, setUrl] = React.useState("");
  return (
    <>
      <h2 className="wiz-h1">Step 1 · Intake</h2>
      <p className="wiz-sub">Validates ticker on EDGAR, fetches entity record, links parent/spinco.</p>

      <div className="field-grid">
        <div className="field">
          <label>Ticker</label>
          <input className="input" value={ticker} onChange={e => setTicker(e.target.value)} />
        </div>
        <div className="field">
          <label>Situation type</label>
          <select className="select" value={situation} onChange={e => setSituation(e.target.value)}>
            <option>Spinoff</option>
            <option>Tracking stock</option>
            <option>Split-off</option>
            <option>Carve-out IPO</option>
          </select>
        </div>
      </div>

      <div className="field" style={{ marginTop: 18 }}>
        <label>Parent ticker (auto-detected)</label>
        <input className="input" value={parent} onChange={e => setParent(e.target.value)} />
      </div>

      <div className="field" style={{ marginTop: 18 }}>
        <label>Optional · newsletter or article URL (seed thesis)</label>
        <input className="input" placeholder="https://…" value={url} onChange={e => setUrl(e.target.value)} />
      </div>

      <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 28 }}>
        <button className="btn primary" onClick={onNext}>Validate and continue {I.arrowR}</button>
      </div>
    </>
  );
}

/* ---------- Step 2: Documents ---------- */
function Step2({ onNext }) {
  return (
    <>
      <h2 className="wiz-h1">Step 2 · Documents</h2>
      <p className="wiz-sub">Add documents three ways: EDGAR auto-fetch, PDF upload, or URL paste.</p>

      <div className="doc-add-grid">
        <div className="doc-add">
          <div className="head"><span className="ico">{I.download}</span> EDGAR fetch</div>
          <div className="desc">Form 10, 10-K, 10-Q, 8-K, DEF 14A</div>
          <button className="btn" style={{ width: "100%", justifyContent: "center" }}>Fetch {I.arrowUR}</button>
        </div>
        <div className="doc-add">
          <div className="head"><span className="ico">{I.up}</span> Upload PDF</div>
          <div className="desc">Decks, paywalled writeups</div>
          <div className="drop-zone">Drop or browse</div>
        </div>
        <div className="doc-add">
          <div className="head"><span className="ico">{I.link}</span> Paste URL</div>
          <div className="desc">Newsletters, transcripts, articles</div>
          <input className="input" placeholder="https://…" style={{ padding: "9px 12px" }} />
        </div>
      </div>

      <h3 className="section-h" style={{ marginBottom: 8 }}>Checklist · 4 of 6 resolved</h3>
      <div className="chk">
        <div className="chk-row">
          <span className="check">{I.check}</span>
          <span>Form 10 · 312 pages</span>
          <span className="pill green">EDGAR</span>
        </div>
        <div className="chk-row">
          <span className="check">{I.check}</span>
          <span>Parent 10-K · 218 pages</span>
          <span className="pill green">EDGAR</span>
        </div>
        <div className="chk-row">
          <span className="check">{I.check}</span>
          <span>Investor day deck</span>
          <span className="pill blue">Uploaded</span>
        </div>
        <div className="chk-row warn">
          <span className="pending">{I.clock}</span>
          <span>Pension footnote · expected May 22</span>
          <span className="pill amber">Pending</span>
        </div>
        <div className="chk-row">
          <span className="empty" />
          <span>Last 2 transcripts · not on EDGAR</span>
          <span className="actions">
            <button className="btn">Upload</button>
            <button className="btn">URL</button>
            <button className="btn">Skip</button>
          </span>
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 28 }}>
        <button className="btn primary" onClick={onNext}>Index documents {I.arrowR}</button>
      </div>
    </>
  );
}

/* ---------- Step 3: Ingest ---------- */
function Step3({ onNext }) {
  return (
    <>
      <h2 className="wiz-h1">Step 3 · Ingest</h2>
      <p className="wiz-sub">LlamaParse + markdown chunking + hybrid retrieval. Failures highlighted in red.</p>

      <div className="ing-list">
        <div className="ing-row">
          <span style={{ color: "var(--green)" }}>{I.check}</span>
          <span>Parsing PDFs · 5 of 5 · 1,847 pages</span>
          <span className="time">42s</span>
        </div>
        <div className="ing-row">
          <span style={{ color: "var(--green)" }}>{I.check}</span>
          <span>Markdown chunking · 3,201 chunks</span>
          <span className="time">18s</span>
        </div>
        <div className="ing-row">
          <span style={{ color: "var(--green)" }}>{I.check}</span>
          <span>Hybrid index · BM25 + embeddings</span>
          <span className="time">31s</span>
        </div>
        <div className="ing-row error">
          <span>{I.alert}</span>
          <div>
            <div>Newsletter URL · 403 forbidden</div>
            <div className="sub">Paywalled. Retry or upload PDF.</div>
          </div>
          <span className="error-actions">
            <button className="btn">Retry</button>
            <button className="btn">Upload</button>
          </span>
        </div>
        <div className="ing-row">
          <span style={{ color: "var(--green)" }}>{I.check}</span>
          <span>Pre-extracted financials</span>
          <span className="time">22s</span>
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 28 }}>
        <button className="btn primary" onClick={onNext}>Explore the corpus {I.arrowR}</button>
      </div>
    </>
  );
}

/* ---------- Step 4: Explore ---------- */
function Step4({ onNext }) {
  const [q, setQ] = React.useState("");
  return (
    <>
      <h2 className="wiz-h1">Step 4 · Explore</h2>
      <p className="wiz-sub">Grounded Q&amp;A with save-to-notes and tags.</p>

      <div className="qa">
        <div className="qa-row">
          <span className="qa-avatar user">{I.user}</span>
          <div className="qa-q">Post-spin net debt and pension exposure?</div>
        </div>
        <div className="qa-row">
          <span className="qa-avatar machine">M</span>
          <div className="qa-a">
            <div className="body">
              $1.2B net debt (3.1x EBITDA). Form 10 p. 87: $340M pension liability. CFO declined to quantify funding shortfall on Q1 call.
            </div>
            <div className="qa-cites">
              <span className="cite">Form 10 p. 87</span>
              <span className="cite">Q1 transcript</span>
              <span className="cite">10-K note 14</span>
            </div>
            <div className="qa-actions">
              <button className="btn">{I.bookmark} Save</button>
              <button className="btn">{I.tag} #pension</button>
              <button className="btn">{I.search} Dig deeper</button>
            </div>
          </div>
        </div>
      </div>

      <div className="qa-input-row">
        <input className="input" placeholder="Ask another question…" value={q} onChange={e => setQ(e.target.value)} />
        <button className="btn primary" onClick={onNext}>Run committee {I.arrowR}</button>
      </div>
    </>
  );
}

/* ---------- Step 5: Committee ---------- */
function Step5({ onNext }) {
  const agents = [
    { k: "S", color: "blue",  who: "Setup specialist",   pts: 82, body: "$1.6B cap excludes 3 indices CTL sits in. Forced selling highly likely." },
    { k: "B", color: "blue",  who: "Business quality",   pts: 84, body: "ROIC 18.4% top-quartile. FCF conversion 92%. Growth 6–8%." },
    { k: "D", color: "red",   who: "Devil's advocate",   pts: null, body: "Pension transfer materially under-disclosed. ~$90M extra funding implied." },
  ];
  return (
    <>
      <h2 className="wiz-h1">Step 5 · Committee</h2>
      <p className="wiz-sub">5 agents debate the Greenblatt scorecard.</p>

      {agents.map((a, i) => (
        <div className="comm-row" key={i}>
          <span className={"comm-avatar"} style={{
            background: a.color === "red" ? "var(--red-soft)" : "var(--blue-soft)",
            color: a.color === "red" ? "var(--red)" : "var(--blue)"
          }}>{a.k}</span>
          <div>
            <div className="who">{a.who}{a.pts != null && <span className="pts"> · {a.pts}</span>}</div>
            <div className="what">{a.body}</div>
          </div>
        </div>
      ))}

      <div className="muted" style={{ marginTop: 14, fontSize: 14 }}>+ Valuation and Incentives agents weighing in…</div>

      <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 24 }}>
        <button className="btn primary" onClick={onNext}>View memo {I.arrowR}</button>
      </div>
    </>
  );
}

/* ---------- Step 6: Memo ---------- */
function Step6({ onDone }) {
  return (
    <>
      <h2 className="wiz-h1">Step 6 · Memo and decision</h2>
      <p className="wiz-sub">Composite 78/100. Click Approve / Watch / Reject.</p>

      <div className="rec-box">
        <div className="label">Recommendation</div>
        <div className="body">
          4 of 5 agents approve. Strong setup, strong business quality, attractive valuation.
          Material concern: pension liability disclosure incomplete. Suggest entry sized 5% pending May 22 footnote.
        </div>
      </div>

      <div className="score-grid">
        <div className="score-tile"><div className="label">Setup</div><div className="val">82</div></div>
        <div className="score-tile"><div className="label">Business</div><div className="val">84</div></div>
        <div className="score-tile"><div className="label">Capital</div><div className="val">65</div></div>
        <div className="score-tile"><div className="label">Valuation</div><div className="val">80</div></div>
        <div className="score-tile"><div className="label">Incentives</div><div className="val">78</div></div>
      </div>

      <div className="decision-row">
        <button className="btn" onClick={onDone}>Reject</button>
        <button className="btn" onClick={onDone}>Watch</button>
        <button className="btn primary" onClick={onDone}>Invest</button>
      </div>
    </>
  );
}

window.Wizard = Wizard;

/* global React, SpinoffUI, MeridianData */
/* =====================================================================
   meridian-thesis.jsx — Thesis-strength surface

   Two new components + three scenario presets:

   - SCENARIOS: three coherent workspace states (researching /
     strengthening / decision-ready) so the user can see how the
     workspace behaves across the score spectrum without having to
     ingest different demo data.

   - ThesisStrengthBar: at-a-glance vertical bar chart of the 5
     buckets with horizontal threshold rulers at 80 (strong) and 90
     (decision-ready). Sits between the workspace header and the
     coach/banner stack — the "drum" of the workspace.

   - ReadyForDecisionBanner: soft nudge. When all 5 ≥ 90, says
     "evidence is strong — review qualitatively and decide." When
     4 of 5 ≥ 80, says "one bucket from strong — chase X."
     Otherwise renders nothing (handing off to the existing coach +
     next-best-action banners).

   Threshold is a SOFT NUDGE per user — never blocks anything, only
   re-orients what the workspace recommends next.
   ===================================================================== */

const { I: IT } = SpinoffUI;
const { DIMS: DIMS_T } = MeridianData;

const STRONG = 80;
const READY  = 90;

const SCENARIOS = {
  researching: {
    key: "researching",
    label: "Researching",
    sub: "Today's LUMN — 3 buckets below ≥80",
    composite: 65,
    state: "researching",
    stateLabel: "Researching",
    coverage: { done: 26, total: 35, blocked: 4 },
    dims: {
      setup:      { score: 88, conf: "High", playedOut: true, coverage: { done: 6, total: 7, blocked: 0 } },
      business:   { score: 82, conf: "High",                  coverage: { done: 5, total: 7, blocked: 0 } },
      capital:    { score: 45, conf: "Low",  blocking: true,  coverage: { done: 5, total: 7, blocked: 1 } },
      valuation:  { score: 60, conf: "Med",                   coverage: { done: 5, total: 7, blocked: 1 } },
      incentives: { score: 48, conf: "Low",  blocking: true,  coverage: { done: 4, total: 7, blocked: 2 } },
    },
  },
  strengthening: {
    key: "strengthening",
    label: "Strengthening",
    sub: "Pension footnote landed · Incentives is the last gap",
    composite: 86,
    state: "researching",
    stateLabel: "Building thesis",
    coverage: { done: 32, total: 35, blocked: 1 },
    dims: {
      setup:      { score: 92, conf: "High", playedOut: true, coverage: { done: 7, total: 7, blocked: 0 } },
      business:   { score: 91, conf: "High",                  coverage: { done: 7, total: 7, blocked: 0 } },
      capital:    { score: 82, conf: "Med",                   coverage: { done: 7, total: 7, blocked: 0 } },
      valuation:  { score: 90, conf: "High",                  coverage: { done: 7, total: 7, blocked: 0 } },
      incentives: { score: 76, conf: "Med",                   coverage: { done: 6, total: 7, blocked: 1 } },
    },
  },
  ready: {
    key: "ready",
    label: "Decision-ready",
    sub: "All 5 buckets ≥ 90 — qualitative + decision unlocked",
    composite: 93,
    state: "memo-ready",
    stateLabel: "Decision-ready",
    coverage: { done: 35, total: 35, blocked: 0 },
    dims: {
      setup:      { score: 92, conf: "High", playedOut: true, coverage: { done: 7, total: 7, blocked: 0 } },
      business:   { score: 94, conf: "High",                  coverage: { done: 7, total: 7, blocked: 0 } },
      capital:    { score: 91, conf: "High",                  coverage: { done: 7, total: 7, blocked: 0 } },
      valuation:  { score: 95, conf: "High",                  coverage: { done: 7, total: 7, blocked: 0 } },
      incentives: { score: 93, conf: "High",                  coverage: { done: 7, total: 7, blocked: 0 } },
    },
  },
};

/* =================================================================
   ThesisStrengthBar — 5 small-multiple bars + threshold ruler
   ================================================================= */
function ThesisStrengthBar({ scenario }) {
  const bars = DIMS_T.map(d => ({
    dim: d,
    score: scenario.dims[d.id].score,
    conf: scenario.dims[d.id].conf,
    playedOut: scenario.dims[d.id].playedOut,
  }));
  const allReady  = bars.every(b => b.score >= READY);
  const allStrong = bars.every(b => b.score >= STRONG);
  const cls = allReady ? "ready" : allStrong ? "strong" : "";

  return (
    <div className={"m-thesis " + cls}>
      <div className="m-thesis-l">
        <div className="m-thesis-label">Thesis strength</div>
        <div className="m-thesis-state">{allReady ? "Decision-ready" : allStrong ? "Strong" : "Building"}</div>
        <div className="m-thesis-composite">
          <span className="v">{scenario.composite}</span>
          <span className="vmax">/100</span>
        </div>
        <div className="m-thesis-meta">{scenario.coverage.done}/{scenario.coverage.total} criteria{scenario.coverage.blocked ? ` · ${scenario.coverage.blocked} blocked` : ""}</div>
      </div>

      <div className="m-thesis-chart">
        <div className="m-thesis-grid">
          <i style={{ bottom: READY  + "%" }} data-mark={READY}  data-label="decision-ready" />
          <i style={{ bottom: STRONG + "%" }} data-mark={STRONG} data-label="strong" />
        </div>
        {bars.map(b => {
          const bcls = b.score >= READY ? "ready" : b.score >= STRONG ? "strong" : "weak";
          return (
            <div key={b.dim.id} className={"m-thesis-col " + b.dim.cls + " " + bcls}>
              <div className="bar-track">
                <div className="bar-fill" style={{ height: Math.max(b.score, 4) + "%" }}>
                  <span className="bar-tip">{b.score}</span>
                </div>
              </div>
              <div className="letter">{b.dim.letter}</div>
            </div>
          );
        })}
      </div>

      <div className="m-thesis-legend">
        <div className="row"><span className="dot ready" /><span>≥ 90 · decision-ready</span></div>
        <div className="row"><span className="dot strong" /><span>≥ 80 · strong</span></div>
        <div className="row"><span className="dot weak" /><span>&lt; 80 · needs work</span></div>
        <div className="row hint">Soft nudge — never blocks the journal.</div>
      </div>
    </div>
  );
}

/* =================================================================
   ReadyForDecisionBanner — soft nudge based on scenario
   ================================================================= */
function ReadyForDecisionBanner({ scenario, onJournal, onMemo }) {
  const all5 = DIMS_T.map(d => scenario.dims[d.id].score);
  const allReady = all5.every(s => s >= READY);
  const allStrong = all5.every(s => s >= STRONG);
  const approaching = !allReady && all5.filter(s => s >= STRONG).length >= 4;
  const laggards = DIMS_T.filter(d => scenario.dims[d.id].score < STRONG);

  if (allReady) {
    return (
      <div className="m-ready-banner ready">
        <span className="ico">{IT.check}</span>
        <div>
          <div className="lbl">Evidence is strong — qualitative review unlocked</div>
          <div className="what">
            All 5 buckets are ≥ {READY}. Capture your qualitative read in the decision journal,
            run the committee one last time, generate the memo, and log the decision.
          </div>
        </div>
        <div className="m-actions">
          <button className="btn ghost" onClick={onMemo}>Generate memo</button>
          <button className="btn pri" onClick={onJournal}>Open decision journal {IT.arrowR}</button>
        </div>
      </div>
    );
  }
  if (approaching) {
    const lag = laggards[0];
    return (
      <div className="m-ready-banner near">
        <span className="ico">{IT.alert}</span>
        <div>
          <div className="lbl">One bucket from strong</div>
          <div className="what">
            4 of 5 buckets are ≥ {STRONG}. {lag ? <><b>{lag.name}</b> at {scenario.dims[lag.id].score} is the gap.</> : null}
            {" "}Decision tools unlock once all 5 reach ≥ {READY}.
          </div>
        </div>
        <div className="m-actions">
          <button className="btn">Why is this gap?</button>
          {lag && <button className="btn pri">Chase {lag.name} docs {IT.arrowR}</button>}
        </div>
      </div>
    );
  }
  return null;
}

window.MeridianThesis = { SCENARIOS, ThesisStrengthBar, ReadyForDecisionBanner, STRONG, READY };

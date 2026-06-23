/* global React, ReactDOM, SpinoffUI, Screens, Wizard */
const { TabBar } = SpinoffUI;
const { Dashboard, Updates, CompanyDetail } = Screens;

function App() {
  const [tab, setTab] = React.useState("dash");

  const openCompany = (_ticker) => setTab("co");
  const goToUpdates = () => setTab("upd");
  const goToDashboard = () => setTab("dash");

  return (
    <div className="app">
      <TabBar tab={tab} setTab={setTab} updates={3} />
      <div data-screen-label={
        tab === "dash" ? "Dashboard" :
        tab === "new"  ? "New analysis" :
        tab === "upd"  ? "Updates" : "Company detail"
      }>
        {tab === "dash" && <Dashboard openCompany={openCompany} goToUpdates={goToUpdates} />}
        {tab === "new"  && <Wizard goToCompany={() => setTab("co")} />}
        {tab === "upd"  && <Updates openCompany={openCompany} />}
        {tab === "co"   && <CompanyDetail />}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

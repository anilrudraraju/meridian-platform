/* global React */
const { useState } = React;

/* ---------- Inline icons (lucide-style) ---------- */
const Icon = ({ d, size = 16, sw = 1.6, fill = "none" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke="currentColor" strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round">
    {d}
  </svg>
);
const I = {
  grid:    <Icon d={<><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></>} />,
  plus:    <Icon d={<><path d="M12 5v14M5 12h14"/></>} />,
  inbox:   <Icon d={<><path d="M22 13l-4 0a2 2 0 0 0-2 2 2 2 0 0 1-2 2h-4a2 2 0 0 1-2-2 2 2 0 0 0-2-2H2"/><path d="M5.5 5h13l3.5 8v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-6l3.5-8z"/></>} />,
  building:<Icon d={<><rect x="4" y="3" width="16" height="18" rx="1.5"/><path d="M9 9h.01M12 9h.01M15 9h.01M9 13h.01M12 13h.01M15 13h.01M10 21v-4h4v4"/></>} />,
  refresh: <Icon d={<><path d="M3 12a9 9 0 0 1 15.6-6.2L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15.6 6.2L3 16"/><path d="M3 21v-5h5"/></>} />,
  download:<Icon d={<><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/></>} />,
  eye:     <Icon d={<><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></>} />,
  copy2:   <Icon d={<><rect x="8" y="8" width="13" height="13" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></>} />,
  chart:   <Icon d={<><path d="M3 3v18h18"/><path d="M7 14l4-4 4 4 5-6"/></>} />,
  line:    <Icon d={<><path d="M3 17l6-6 4 4 8-8"/></>} />,
  check2:  <Icon d={<><path d="M3 7l4 4M14 4l-4 12-6-6"/><path d="M21 6l-9 9"/></>} />,
  folder:  <Icon d={<><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/></>} />,
  bookmark:<Icon d={<><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></>} />,
  chat:    <Icon d={<><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></>} />,
  bell:    <Icon d={<><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10 21a2 2 0 0 0 4 0"/></>} />,
  clock:   <Icon d={<><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>} />,
  doc:     <Icon d={<><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></>} />,
  check:   <Icon d={<><path d="M5 12l4 4 10-10"/></>} />,
  alert:   <Icon d={<><path d="M10.3 3.86l-8.1 14a2 2 0 0 0 1.7 3h16.2a2 2 0 0 0 1.7-3l-8.1-14a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/></>} />,
  down:    <Icon d={<><path d="M7 13l5 5 5-5"/><path d="M12 18V4"/></>} />,
  up:      <Icon d={<><path d="M17 11l-5-5-5 5"/><path d="M12 6v14"/></>} />,
  link:    <Icon d={<><path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1"/><path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1"/></>} />,
  user:    <Icon d={<><circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/></>} />,
  tag:     <Icon d={<><path d="M20 12l-8.6 8.6a2 2 0 0 1-2.8 0L2 13.9V4h9.9z"/><circle cx="7.5" cy="8.5" r="1.2"/></>} />,
  search:  <Icon d={<><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></>} />,
  arrowR:  <Icon d={<><path d="M5 12h14M13 6l6 6-6 6"/></>} />,
  arrowUR: <Icon d={<><path d="M7 17 17 7M9 7h8v8"/></>} />,
  arrowD:  <Icon d={<><path d="M12 5v14M6 13l6 6 6-6"/></>} />,
  kebab:   <Icon d={<><circle cx="12" cy="5" r="1.4" fill="currentColor"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/><circle cx="12" cy="19" r="1.4" fill="currentColor"/></>} />,
};

/* ---------- Tab bar ---------- */
function TabBar({ tab, setTab, updates }) {
  const tabs = [
    { id: "dash", label: "Dashboard", ico: I.grid },
    { id: "new",  label: "New analysis", ico: I.plus },
    { id: "upd",  label: "Updates", ico: I.inbox, badge: updates },
    { id: "co",   label: "Company detail", ico: I.building },
  ];
  return (
    <div className="tabs">
      {tabs.map(t => (
        <button key={t.id} className={"tab " + (tab === t.id ? "active" : "")} onClick={() => setTab(t.id)}>
          <span className="ico">{t.ico}</span>
          <span>{t.label}</span>
          {t.badge ? <span className="badge">{t.badge}</span> : null}
        </button>
      ))}
      <div className="tab-spacer" />
      {(tab === "upd" || tab === "co" || tab === "new") && (
        <button className="tab-menu" aria-label="More">{I.kebab}</button>
      )}
    </div>
  );
}

/* ---------- Dimension dots renderer ---------- */
function Dims({ vals }) {
  // vals: array of "g"/"a"/"r"/"x"
  return (
    <span className="dims">
      {vals.map((v, i) => <span key={i} className={"dim " + v} />)}
    </span>
  );
}

window.SpinoffUI = { Icon, I, TabBar, Dims };
window.useState = useState;

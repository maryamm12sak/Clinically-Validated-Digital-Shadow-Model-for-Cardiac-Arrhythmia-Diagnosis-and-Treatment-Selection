
import { useState } from "react";
import { Link, NavLink, Route, Routes } from "react-router-dom";
import { LatestProvider, MetaProvider, useMeta } from "./state.jsx";
import About from "./pages/About.jsx";
import Analyze from "./pages/Analyze.jsx";
import AnalysisDetail from "./pages/AnalysisDetail.jsx";
import History from "./pages/History.jsx";
import Results from "./pages/Results.jsx";
import Samples from "./pages/Samples.jsx";

function StatusPill() {
  const { health } = useMeta();
  if (health.state === "checking") return <span className="status status-checking">Checking server…</span>;
  if (health.state === "offline") return <span className="status status-offline">Server offline</span>;
  return (
    <span className="status status-online">
      Server online · Model: {health.model_loaded ? (health.model_version ?? "loaded") : "placeholder"}
    </span>
  );
}

function Header({ open, onToggle }) {
  return (
    <header className="topbar">
      <button
        type="button"
        className="menu-btn"
        onClick={onToggle}
        aria-label={open ? "Close menu" : "Open menu"}
        aria-expanded={open}
      >
        ☰
      </button>
      <Link to="/" className="brand">
        <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden="true">
          <rect width="32" height="32" rx="7" fill="#fff" fillOpacity=".18" />
          <path d="M3 17h7l3-8 5 15 3-9 2 2h6" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        ECG Decision Support
      </Link>
      <StatusPill />
    </header>
  );
}

function Sidebar({ open }) {
  return (
    <aside className={`sidebar ${open ? "" : "closed"}`} aria-label="Main navigation">
      <nav>
        <NavLink to="/" end>Analyse</NavLink>
        <NavLink to="/samples">Demo ECGs</NavLink>
        <NavLink to="/history">History</NavLink>
        <NavLink to="/about">About</NavLink>
      </nav>
    </aside>
  );
}

function OfflineBanner() {
  const { health } = useMeta();
  if (health.state !== "offline") return null;
  return (
    <div className="alert alert-red offline-banner" role="alert">
      <strong>Can't reach the backend.</strong> Start it with <code>uvicorn main:app --reload</code> and check{" "}
      <code>VITE_API_BASE_URL</code> in <code>.env</code>.
    </div>
  );
}

function Footer() {
  const { disclaimer } = useMeta();
  return <footer className="footer">{disclaimer}</footer>;
}

function NotFound() {
  return (
    <div className="stack">
      <h1>Page not found</h1>
      <Link to="/">Back to Analyse</Link>
    </div>
  );
}

export default function App() {
  const [open, setOpen] = useState(() => {
    try {
      const saved = localStorage.getItem("ecg.sidebar");
      if (saved) return saved === "open";
    } catch { /* ignore */ }
    return window.innerWidth > 860; // closed by default on phones
  });
  const toggle = () =>
    setOpen((o) => {
      try { localStorage.setItem("ecg.sidebar", o ? "closed" : "open"); } catch { /* ignore */ }
      return !o;
    });

  return (
    <MetaProvider>
      <LatestProvider>
        <Header open={open} onToggle={toggle} />
        <div className="shell">
          <Sidebar open={open} />
          <div className="content">
            <main className="container">
              <OfflineBanner />
              <Routes>
                <Route path="/" element={<Analyze />} />
                <Route path="/results/:id" element={<Results />} />
                <Route path="/samples" element={<Samples />} />
                <Route path="/history" element={<History />} />
                <Route path="/history/:id" element={<AnalysisDetail />} />
                <Route path="/about" element={<About />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </main>
            <Footer />
          </div>
        </div>
      </LatestProvider>
    </MetaProvider>
  );
}
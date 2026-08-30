import { useEffect, useState } from "react";
import Toast from "../components/Toast.jsx";
import { AttackResults, ListAttacks } from "../services/api.js";

export default function AttackLab() {
  const [results, setResults] = useState(null);
  const [catalog, setCatalog] = useState([]);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    ListAttacks().then((d) => setCatalog(d.attacks));
    AttackResults()
      .then(setResults)
      .catch((e) => setError(e?.response?.data?.detail || e.message));
  }, []);

  const catalogByType = Object.fromEntries(catalog.map((c) => [c.attack_type, c]));
  const generated = results ? new Set(results.attacks.map((a) => a.attack_type)) : new Set();

  return (
    <div>
      <h1 className="page-title">Attack Lab</h1>
      <p className="page-subtitle">All 20 registered attack strategies. Cards show real generation/detection numbers once you've generated that attack type.</p>

      <div className="grid grid-3">
        {catalog.map((c) => {
          const stats = results?.attacks.find((a) => a.attack_type === c.attack_type);
          return (
            <div
              key={c.attack_type}
              className="card"
              style={{ cursor: "pointer", borderColor: selected === c.attack_type ? "var(--accent)" : undefined }}
              onClick={() => setSelected(c.attack_type)}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ fontWeight: 800, fontSize: 13 }}>{c.attack_type}</div>
                <span className={`badge ${c.severity.toLowerCase()}`}>{c.severity}</span>
              </div>
              <p style={{ fontSize: 12, color: "var(--text-dim)", minHeight: 48 }}>{c.description}</p>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                <span>Generated: <b>{stats ? stats.n_generated : 0}</b></span>
                <span>Detection: <b>{stats && stats.detection_rate != null ? `${(stats.detection_rate * 100).toFixed(0)}%` : "–"}</b></span>
              </div>
            </div>
          );
        })}
      </div>

      {selected && catalogByType[selected] && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="section-heading">{selected} -- details</div>
          <p style={{ fontSize: 13.5, color: "var(--text-dim)" }}>{catalogByType[selected].description}</p>
          <p style={{ fontSize: 12.5 }}>
            <b>Features affected:</b> {catalogByType[selected].features_affected.join(", ")}
          </p>
          {results?.attacks.find((a) => a.attack_type === selected) ? (
            (() => {
              const s = results.attacks.find((a) => a.attack_type === selected);
              return (
                <div className="grid grid-3" style={{ marginTop: 12 }}>
                  <Mini label="Generated" value={s.n_generated} />
                  <Mini label="Avg Difficulty" value={s.avg_difficulty.toFixed(2)} />
                  <Mini label="Detection Rate" value={s.detection_rate != null ? `${(s.detection_rate * 100).toFixed(1)}%` : "no model yet"} />
                </div>
              );
            })()
          ) : (
            <p style={{ color: "var(--text-dim)", fontSize: 12.5 }}>
              Not generated yet -- head to Red Team → Attack Generator to create some.
            </p>
          )}
        </div>
      )}

      <Toast message={error} type="error" onClose={() => setError(null)} />
    </div>
  );
}

function Mini({ label, value }) {
  return (
    <div>
      <div style={{ fontSize: 10.5, color: "var(--text-dim)", textTransform: "uppercase", fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 800 }}>{value}</div>
    </div>
  );
}

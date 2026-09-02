import { useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import Toast from "../components/Toast.jsx";
import { RunClosedLoop } from "../services/api.js";

export default function ClosedLoop() {
  const [iterations, setIterations] = useState(3);
  const [nPerType, setNPerType] = useState(150);
  const [useLlm, setUseLlm] = useState(true);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      const r = await RunClosedLoop(Number(iterations), Number(nPerType), useLlm, 42);
      setResult(r);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setRunning(false);
    }
  };

  const chartData = result
    ? result.iterations.map((it) => ({
        iteration: `Iter ${it.iteration}`,
        recall: it.after_metrics.recall,
        precision: it.after_metrics.precision,
        f1: it.after_metrics.f1,
      }))
    : [];

  return (
    <div>
      <h1 className="page-title">Closed Loop</h1>
      <p className="page-subtitle">
        Generate → Detect → Analyze Failure → Generate Harder Attack → Retrain → Improve. Every metric below is a real
        measured before/after on held-out hard examples, never hard-coded.
      </p>

      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: "flex", gap: 18, alignItems: "flex-end", flexWrap: "wrap" }}>
          <div className="field" style={{ marginBottom: 0 }}>
            <label>Iterations</label>
            <input type="number" min="1" max="8" value={iterations} onChange={(e) => setIterations(e.target.value)} style={{ width: 90 }} />
          </div>
          <div className="field" style={{ marginBottom: 0 }}>
            <label>New hard attacks / hypothesis</label>
            <input type="number" min="10" max="2000" value={nPerType} onChange={(e) => setNPerType(e.target.value)} style={{ width: 110 }} />
          </div>
          <div className="field" style={{ marginBottom: 0 }}>
            <label>Attack discovery</label>
            <div className="pill-row">
              <span className={`pill ${useLlm ? "active" : ""}`} onClick={() => setUseLlm(true)} style={{ cursor: "pointer" }}>
                Local LLM (Ollama, if running)
              </span>
              <span className={`pill ${!useLlm ? "active" : ""}`} onClick={() => setUseLlm(false)} style={{ cursor: "pointer" }}>
                Rule-based only
              </span>
            </div>
          </div>
          <button className="btn" onClick={run} disabled={running}>
            {running ? "Running loop…" : "RUN CLOSED LOOP"}
          </button>
        </div>
      </div>

      {result && (
        <>
          <div className="card" style={{ marginBottom: 20 }}>
            <div className="section-heading">Recall / Precision / F1 vs iteration</div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#232c3d" />
                <XAxis dataKey="iteration" stroke="#8b97ac" fontSize={11} />
                <YAxis stroke="#8b97ac" fontSize={11} domain={[0, 1]} />
                <Tooltip contentStyle={{ background: "#131a26", border: "1px solid #232c3d" }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="recall" stroke="#4f8cff" strokeWidth={2} />
                <Line type="monotone" dataKey="precision" stroke="#7c5cff" strokeWidth={2} />
                <Line type="monotone" dataKey="f1" stroke="#35d08f" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {result.iterations.map((it) => (
            <div className="card" key={it.iteration} style={{ marginBottom: 16 }}>
              <div className="section-heading">Iteration {it.iteration} → model_v{it.model_version}</div>
              {it.weakness_summary && (
                <p style={{ fontSize: 12.5, color: "var(--text-dim)", background: "#0e1420", padding: 10, borderRadius: 8 }}>
                  {it.weakness_summary}
                </p>
              )}
              <div className="grid grid-4" style={{ marginTop: 10 }}>
                <BeforeAfter label="Recall" before={it.before_metrics?.recall} after={it.after_metrics.recall} />
                <BeforeAfter label="Precision" before={it.before_metrics?.precision} after={it.after_metrics.precision} />
                <BeforeAfter label="F1" before={it.before_metrics?.f1} after={it.after_metrics.f1} />
                <div>
                  <div style={{ fontSize: 10.5, color: "var(--text-dim)", textTransform: "uppercase", fontWeight: 700 }}>New hard attacks</div>
                  <div style={{ fontSize: 20, fontWeight: 800 }}>{it.new_attack_count}</div>
                </div>
              </div>
            </div>
          ))}
        </>
      )}

      <Toast message={error} type="error" onClose={() => setError(null)} />
    </div>
  );
}

function BeforeAfter({ label, before, after }) {
  const improved = before != null && after >= before;
  return (
    <div>
      <div style={{ fontSize: 10.5, color: "var(--text-dim)", textTransform: "uppercase", fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 800 }}>
        {before != null ? `${(before * 100).toFixed(1)}%` : "–"}
        <span style={{ color: "var(--text-dim)", fontWeight: 500, margin: "0 6px" }}>→</span>
        <span style={{ color: improved ? "var(--ok)" : "var(--danger)" }}>{(after * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
}

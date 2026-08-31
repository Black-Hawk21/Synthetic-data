import { useEffect, useState } from "react";
import StatCard from "../components/StatCard.jsx";
import Toast from "../components/Toast.jsx";
import { DatasetSummary, Metrics, GraphRings, GenerateApplicants } from "../services/api.js";

export default function Overview() {
  const [summary, setSummary] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [rings, setRings] = useState(null);
  const [error, setError] = useState(null);
  const [bootstrapping, setBootstrapping] = useState(false);

  const load = async () => {
    try {
      const [s, m, r] = await Promise.all([
        DatasetSummary(),
        Metrics().catch(() => null),
        GraphRings(3).catch(() => null),
      ]);
      setSummary(s);
      setMetrics(m);
      setRings(r);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const bootstrap = async () => {
    setBootstrapping(true);
    setError(null);
    try {
      await GenerateApplicants(2000, 1);
      await load();
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setBootstrapping(false);
    }
  };

  const noData = summary && summary.total_applicants === 0;

  return (
    <div>
      <h1 className="page-title">Identity Fraud Defense Lab</h1>
      <p className="page-subtitle">
        Closed-loop red team / blue team system for GenAI-powered identity &amp; onboarding fraud.
        Every number on this page comes from the live FastAPI backend -- nothing here is hard-coded.
      </p>

      {noData && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="section-heading">No dataset yet</div>
          <p style={{ color: "var(--text-dim)", fontSize: 13.5 }}>
            Generate a starter population of legitimate applicants to begin, or head to{" "}
            <b>Red Team → Attack Generator</b> to start from an attack.
          </p>
          <button className="btn" onClick={bootstrap} disabled={bootstrapping}>
            {bootstrapping ? "Generating…" : "Generate 2,000 legitimate applicants"}
          </button>
        </div>
      )}

      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <StatCard label="Total Synthetic Applicants" value={summary ? summary.total_applicants.toLocaleString() : "–"} />
        <StatCard
          label="Total Attacks Generated"
          value={summary ? summary.total_fraud.toLocaleString() : "–"}
          sub={summary ? `${(summary.fraud_rate * 100).toFixed(1)}% of dataset` : ""}
        />
        <StatCard label="Attack Types Present" value={summary ? Object.keys(summary.attack_types).length : "–"} sub="of 20 registered" />
        <StatCard label="Current Model Version" value={metrics?.model_version ?? "–"} sub={metrics?.model_version ? "trained" : "not trained yet"} />
      </div>

      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <StatCard
          label="Detection Recall"
          value={metrics?.model_metrics ? `${(metrics.model_metrics.final_model_metrics.recall * 100).toFixed(1)}%` : "–"}
        />
        <StatCard
          label="False Positive Rate"
          value={metrics?.model_metrics ? `${(metrics.model_metrics.final_model_metrics.false_positive_rate * 100).toFixed(1)}%` : "–"}
        />
        <StatCard
          label="PR-AUC"
          value={metrics?.model_metrics ? metrics.model_metrics.final_model_metrics.pr_auc.toFixed(3) : "–"}
        />
        <StatCard label="Fraud Rings Discovered" value={rings ? rings.num_clusters : "–"} sub={rings ? `${rings.total_identities_in_clusters} identities involved` : ""} />
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="section-heading">Closed-loop architecture</div>
        <div className="loop-flow">
          <span className="step">IDENTIFY</span>
          <span className="arrow">→</span>
          <span className="step">GENERATE</span>
          <span className="arrow">→</span>
          <span className="step">DETECT</span>
          <span className="arrow">→</span>
          <span className="step">ANALYZE FAILURE</span>
          <span className="arrow">→</span>
          <span className="step">HARDEN &amp; RETRAIN</span>
          <span className="arrow">→</span>
          <span className="step">LEARN</span>
          <span className="arrow">↻ back to GENERATE</span>
        </div>
        <p style={{ color: "var(--text-dim)", fontSize: 13, marginTop: 14 }}>
          Red Team generates synthetic identity/onboarding fraud across 20 attack types. Blue Team scores every
          applicant with an XGBoost detector trained on real, generated data. The Feedback Loop finds what the
          detector missed, generates harder variants of exactly those attacks, and retrains -- see the{" "}
          <b>Closed Loop</b> page for measured before/after recall.
        </p>
      </div>

      {summary && Object.keys(summary.attack_types).length > 0 && (
        <div className="card">
          <div className="section-heading">Attack type distribution</div>
          <table>
            <thead>
              <tr>
                <th>Attack Type</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(summary.attack_types)
                .sort((a, b) => b[1] - a[1])
                .map(([type, count]) => (
                  <tr key={type}>
                    <td className="mono">{type}</td>
                    <td>{count}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      <Toast message={error} type="error" onClose={() => setError(null)} />
    </div>
  );
}

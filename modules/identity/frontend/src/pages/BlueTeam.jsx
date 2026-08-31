import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import Toast from "../components/Toast.jsx";
import { DecisionBadge } from "../components/Badges.jsx";
import { RunBlueTeam, RunDetection, ScoreApplicant } from "../services/api.js";

const METRIC_KEYS = ["precision", "recall", "f1", "pr_auc"];

export default function BlueTeam() {
  const [training, setTraining] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [trainResult, setTrainResult] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [threshold, setThreshold] = useState(0.7);
  const [applicantId, setApplicantId] = useState("");
  const [scoreResult, setScoreResult] = useState(null);
  const [scoring, setScoring] = useState(false);
  const [error, setError] = useState(null);

  const runBlueTeam = async () => {
    setTraining(true);
    setError(null);
    try {
      setTrainResult(await RunBlueTeam(42));
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setTraining(false);
    }
  };

  const runDetection = async () => {
    setDetecting(true);
    setError(null);
    try {
      setDetectionResult(await RunDetection(Number(threshold)));
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setDetecting(false);
    }
  };

  const score = async () => {
    if (!applicantId) return;
    setScoring(true);
    setError(null);
    try {
      setScoreResult(await ScoreApplicant(applicantId, 6));
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setScoring(false);
    }
  };

  const comparisonChart = trainResult
    ? Object.entries(trainResult.comparison_metrics).map(([model, m]) => ({
        model,
        ...Object.fromEntries(METRIC_KEYS.map((k) => [k, m[k]])),
      }))
    : [];

  return (
    <div>
      <h1 className="page-title">Blue Team · Detection</h1>
      <p className="page-subtitle">Train the detector, run batch detection, and score individual applicants -- all live model calls.</p>

      <div className="grid grid-2" style={{ marginBottom: 20, alignItems: "start" }}>
        <div className="card">
          <div className="section-heading">1. Train detector</div>
          <p style={{ fontSize: 12.5, color: "var(--text-dim)" }}>
            Trains Logistic Regression, Random Forest and XGBoost on the current dataset; XGBoost is saved as the
            active model version.
          </p>
          <button className="btn" onClick={runBlueTeam} disabled={training}>
            {training ? "Training…" : "RUN BLUE TEAM"}
          </button>

          {trainResult && (
            <div style={{ marginTop: 18 }}>
              <p style={{ fontSize: 12.5 }}>Saved as <b>model_v{trainResult.model_version}</b> ({trainResult.n_trained_on.toLocaleString()} rows)</p>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={comparisonChart}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#232c3d" />
                  <XAxis dataKey="model" stroke="#8b97ac" fontSize={11} />
                  <YAxis stroke="#8b97ac" fontSize={11} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: "#131a26", border: "1px solid #232c3d" }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="precision" fill="#4f8cff" />
                  <Bar dataKey="recall" fill="#7c5cff" />
                  <Bar dataKey="f1" fill="#35d08f" />
                  <Bar dataKey="pr_auc" fill="#ffb454" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        <div className="card">
          <div className="section-heading">2. Run detection on full dataset</div>
          <div className="field">
            <label>Decision threshold: {Number(threshold).toFixed(2)}</label>
            <input type="range" min="0" max="1" step="0.05" value={threshold} onChange={(e) => setThreshold(e.target.value)} />
          </div>
          <button className="btn" onClick={runDetection} disabled={detecting}>
            {detecting ? "Scoring…" : "RUN DETECTION"}
          </button>

          {detectionResult && (
            <div style={{ marginTop: 16 }}>
              <p style={{ fontSize: 15, fontWeight: 700 }}>
                Detected: {detectionResult.detected} / {detectionResult.total_fraud}
              </p>
              <div className="grid grid-4">
                <MetricMini label="Precision" value={detectionResult.metrics.precision} />
                <MetricMini label="Recall" value={detectionResult.metrics.recall} />
                <MetricMini label="F1" value={detectionResult.metrics.f1} />
                <MetricMini label="PR-AUC" value={detectionResult.metrics.pr_auc} />
                <MetricMini label="FPR" value={detectionResult.metrics.false_positive_rate} />
                <MetricMini label="FNR" value={detectionResult.metrics.false_negative_rate} />
                <MetricMini label="ROC-AUC" value={detectionResult.metrics.roc_auc} />
              </div>
            </div>
          )}
        </div>
      </div>

      {detectionResult && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="section-heading">Recall by attack type</div>
          <table>
            <thead><tr><th>Attack Type</th><th>Recall</th><th>Samples</th></tr></thead>
            <tbody>
              {Object.entries(detectionResult.per_attack_type_recall)
                .sort((a, b) => a[1].recall - b[1].recall)
                .map(([type, stats]) => (
                  <tr key={type}>
                    <td className="mono">{type}</td>
                    <td>{(stats.recall * 100).toFixed(1)}%</td>
                    <td>{stats.n_samples}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="card">
        <div className="section-heading">3. Score a single applicant</div>
        <div style={{ display: "flex", gap: 10, marginBottom: 14 }}>
          <input type="text" placeholder="APP_xxxxxxxxxxxxxxxx" value={applicantId} onChange={(e) => setApplicantId(e.target.value)} style={{ flex: 1 }} />
          <button className="btn secondary" onClick={score} disabled={scoring}>
            {scoring ? "Scoring…" : "Score"}
          </button>
        </div>
        {scoreResult && (
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 10 }}>
              <div style={{ fontSize: 28, fontWeight: 800 }}>{scoreResult.risk_score}% RISK</div>
              <DecisionBadge decision={scoreResult.decision} />
            </div>
            {scoreResult.risk_factors.map((f, i) => (
              <div className="risk-factor" key={i}><span>{i + 1}. {f.description}</span></div>
            ))}
          </div>
        )}
      </div>

      <Toast message={error} type="error" onClose={() => setError(null)} />
    </div>
  );
}

function MetricMini({ label, value }) {
  return (
    <div>
      <div style={{ fontSize: 10.5, color: "var(--text-dim)", textTransform: "uppercase", fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 800 }}>{value != null ? value.toFixed(3) : "–"}</div>
    </div>
  );
}

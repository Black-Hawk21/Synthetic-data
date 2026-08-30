import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import Toast from "../components/Toast.jsx";
import { SeverityBadge } from "../components/Badges.jsx";
import { ListAttacks, GenerateAttack, DatasetSummary } from "../services/api.js";

export default function RedTeam() {
  const [attacks, setAttacks] = useState([]);
  const [attackType, setAttackType] = useState("FRAUD_RING");
  const [difficulty, setDifficulty] = useState(0.5);
  const [n, setN] = useState(200);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    ListAttacks().then((d) => setAttacks(d.attacks));
    DatasetSummary().then(setSummary);
  }, []);

  const selected = attacks.find((a) => a.attack_type === attackType);

  const generate = async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await GenerateAttack(attackType, Number(difficulty), Number(n));
      setResult(r);
      setSummary(await DatasetSummary());
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const chartData = summary
    ? Object.entries(summary.attack_types)
        .sort((a, b) => b[1] - a[1])
        .map(([type, count]) => ({ type, count }))
    : [];

  return (
    <div>
      <h1 className="page-title">Red Team · Attack Generator</h1>
      <p className="page-subtitle">Generate realistic synthetic identity/onboarding fraud attacks at scale.</p>

      <div className="grid" style={{ gridTemplateColumns: "340px 1fr", gap: 20, alignItems: "start" }}>
        <div className="card">
          <div className="section-heading">Configure attack</div>
          <div className="field">
            <label>Attack Type</label>
            <select value={attackType} onChange={(e) => setAttackType(e.target.value)}>
              {attacks.map((a) => (
                <option key={a.attack_type} value={a.attack_type}>
                  {a.attack_type}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Difficulty: {Number(difficulty).toFixed(2)} ({difficulty < 0.34 ? "Easy" : difficulty < 0.7 ? "Medium" : "Hard"})</label>
            <input type="range" min="0" max="1" step="0.05" value={difficulty} onChange={(e) => setDifficulty(e.target.value)} />
          </div>
          <div className="field">
            <label>Number of Applications</label>
            <input type="number" min="1" max="200000" value={n} onChange={(e) => setN(e.target.value)} />
          </div>
          {selected && (
            <p style={{ fontSize: 12.5, color: "var(--text-dim)", marginBottom: 14 }}>{selected.description}</p>
          )}
          <button className="btn" onClick={generate} disabled={loading}>
            {loading ? "Generating…" : "GENERATE ATTACK"}
          </button>
        </div>

        <div>
          {result && (
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="section-heading">
                {result.attack_type} <SeverityBadge severity={result.severity} />
              </div>
              <p style={{ fontSize: 13.5, color: "var(--text-dim)" }}>{result.description}</p>
              <div className="grid grid-4">
                <StatMini label="Generated" value={result.n_generated} />
                <StatMini label="Suspicious Clusters" value={result.suspicious_clusters} />
                <StatMini label="Infra Reuse Cases" value={result.infra_reuse_cases} />
                <StatMini label="Total Dataset Size" value={result.total_dataset_size} />
              </div>
            </div>
          )}

          <div className="card">
            <div className="section-heading">Current attack distribution</div>
            {chartData.length === 0 ? (
              <div className="empty-state">No attacks generated yet.</div>
            ) : (
              <ResponsiveContainer width="100%" height={Math.max(220, chartData.length * 28)}>
                <BarChart data={chartData} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#232c3d" horizontal={false} />
                  <XAxis type="number" stroke="#8b97ac" fontSize={11} />
                  <YAxis type="category" dataKey="type" stroke="#8b97ac" fontSize={10.5} width={190} />
                  <Tooltip contentStyle={{ background: "#131a26", border: "1px solid #232c3d" }} />
                  <Bar dataKey="count" fill="#4f8cff" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>

      <Toast message={error} type="error" onClose={() => setError(null)} />
    </div>
  );
}

function StatMini({ label, value }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--text-dim)", textTransform: "uppercase", fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 800 }}>{value}</div>
    </div>
  );
}

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Toast from "../components/Toast.jsx";
import { DecisionBadge } from "../components/Badges.jsx";
import { ListAttacks, SimulateApplicant, ScoreApplicant } from "../services/api.js";

const SECTIONS = [
  {
    title: "Identity",
    fields: ["name", "age", "date_of_birth", "country", "city", "phone", "email", "identity_age_days", "previous_application_count"],
  },
  {
    title: "Document",
    fields: ["document_type", "document_quality", "ocr_confidence", "doc_field_consistency", "document_authenticity_score", "document_tamper_score", "name_match_score", "dob_match_score"],
  },
  {
    title: "Face / Biometric",
    fields: ["face_similarity_score", "liveness_score", "deepfake_probability", "face_quality_score", "face_reuse_count"],
  },
  {
    title: "Device",
    fields: ["device_id", "device_age_days", "device_reuse_count", "os", "browser", "timezone"],
  },
  {
    title: "Network",
    fields: ["ip_id", "ip_reuse_count", "identities_from_ip", "geo_consistency", "asn_category", "vpn_proxy_probability"],
  },
  {
    title: "Behavior",
    fields: ["session_duration_sec", "form_completion_time_sec", "automation_score", "typing_variance", "mouse_entropy", "application_velocity"],
  },
];

function fmt(v) {
  if (typeof v === "number") return Number.isInteger(v) ? v : v.toFixed(3);
  return String(v);
}

export default function OnboardingSimulator() {
  const [attacks, setAttacks] = useState([]);
  const [attackType, setAttackType] = useState("NONE");
  const [difficulty, setDifficulty] = useState(0.5);
  const [applicant, setApplicant] = useState(null);
  const [verification, setVerification] = useState(null);
  const [loading, setLoading] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    ListAttacks().then((d) => setAttacks(d.attacks));
  }, []);

  const generate = async () => {
    setLoading(true);
    setError(null);
    setVerification(null);
    try {
      const a = await SimulateApplicant(attackType === "NONE" ? null : attackType, Number(difficulty));
      setApplicant(a);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const runVerification = async () => {
    if (!applicant) return;
    setVerifying(true);
    setError(null);
    try {
      const v = await ScoreApplicant(applicant.applicant_id, 6);
      setVerification(v);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setVerifying(false);
    }
  };

  return (
    <div>
      <h1 className="page-title">Onboarding Simulator</h1>
      <p className="page-subtitle">Generate a synthetic applicant, then run it through the live Blue Team detector.</p>

      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: "flex", gap: 14, alignItems: "flex-end", flexWrap: "wrap" }}>
          <div className="field" style={{ marginBottom: 0, minWidth: 240 }}>
            <label>Scenario</label>
            <select value={attackType} onChange={(e) => setAttackType(e.target.value)}>
              <option value="NONE">Legitimate applicant</option>
              {attacks.map((a) => (
                <option key={a.attack_type} value={a.attack_type}>
                  {a.attack_type}
                </option>
              ))}
            </select>
          </div>
          {attackType !== "NONE" && (
            <div className="field" style={{ marginBottom: 0, minWidth: 200 }}>
              <label>Difficulty: {Number(difficulty).toFixed(2)}</label>
              <input type="range" min="0" max="1" step="0.05" value={difficulty} onChange={(e) => setDifficulty(e.target.value)} />
            </div>
          )}
          <button className="btn" onClick={generate} disabled={loading}>
            {loading ? "Generating…" : "Generate Applicant"}
          </button>
          {applicant && (
            <button className="btn secondary" onClick={runVerification} disabled={verifying}>
              {verifying ? "Verifying…" : "RUN VERIFICATION"}
            </button>
          )}
        </div>
      </div>

      {applicant && (
        <div className="grid grid-3" style={{ marginBottom: 20 }}>
          {SECTIONS.map((sec) => (
            <div className="card" key={sec.title}>
              <div className="section-heading">{sec.title}</div>
              {sec.fields.map((f) => (
                <div key={f} className="risk-factor">
                  <span style={{ color: "var(--text-dim)" }}>{f.replaceAll("_", " ")}</span>
                  <span className="mono">{applicant[f] !== undefined ? fmt(applicant[f]) : "–"}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}

      {verification && (
        <div className="card">
          <div className="section-heading">Verification Result</div>
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 14 }}>
            <div style={{ fontSize: 34, fontWeight: 800 }}>{verification.risk_score}% RISK</div>
            <DecisionBadge decision={verification.decision} />
            <Link to="/graph" style={{ marginLeft: "auto", fontSize: 12.5, color: "var(--accent)" }}>
              View in Identity Graph →
            </Link>
          </div>
          <div className="section-heading" style={{ fontSize: 13 }}>Top Risk Factors</div>
          {verification.risk_factors.length === 0 ? (
            <div className="empty-state">No elevated risk factors -- looks clean.</div>
          ) : (
            verification.risk_factors.map((f, i) => (
              <div className="risk-factor" key={i}>
                <span>{i + 1}. {f.description}</span>
              </div>
            ))
          )}
          <p style={{ color: "var(--text-dim)", fontSize: 12, marginTop: 12 }}>
            Model v{verification.model_version} · Ground truth: {verification.ground_truth.is_fraud ? verification.ground_truth.attack_type : "legitimate"}
          </p>
        </div>
      )}

      <Toast message={error} type="error" onClose={() => setError(null)} />
    </div>
  );
}

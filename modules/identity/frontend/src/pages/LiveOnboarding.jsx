import { useState } from "react";
import Toast from "../components/Toast.jsx";
import WebcamCapture from "../components/WebcamCapture.jsx";
import { DecisionBadge } from "../components/Badges.jsx";
import { useDeviceFingerprint } from "../hooks/useDeviceFingerprint.js";
import { useTelemetry } from "../hooks/useTelemetry.js";
import { SampleDocumentUrl, SubmitManualOnboarding } from "../services/api.js";

const EMPTY_FORM = { name: "", date_of_birth: "", address: "", phone: "", email: "", document_type: "NATIONAL_ID", document_number: "" };

export default function LiveOnboarding() {
  const [form, setForm] = useState(EMPTY_FORM);
  const [documentFile, setDocumentFile] = useState(null);
  const [documentPreview, setDocumentPreview] = useState(null);
  const [selfieBlob, setSelfieBlob] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const fingerprint = useDeviceFingerprint();
  const { onKeyDown, onFieldFocus, onFieldBlur, snapshot } = useTelemetry();

  const set = (key) => (e) => setForm((f) => ({ ...f, [key]: e.target.value }));

  const onDocumentChange = (e) => {
    const file = e.target.files?.[0];
    setDocumentFile(file || null);
    setDocumentPreview(file ? URL.createObjectURL(file) : null);
  };

  const submit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const telemetry = snapshot();
      const r = await SubmitManualOnboarding(form, documentFile, selfieBlob, telemetry, fingerprint || {});
      setResult(r);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message);
    } finally {
      setSubmitting(false);
    }
  };

  const reset = () => {
    setForm(EMPTY_FORM);
    setDocumentFile(null);
    setDocumentPreview(null);
    setSelfieBlob(null);
    setResult(null);
  };

  return (
    <div onKeyDown={onKeyDown}>
      <h1 className="page-title">Live KYC Form</h1>
      <p className="page-subtitle">
        A real onboarding form -- type your details, upload a document photo, capture a live selfie. The exact same
        form works for a legitimate signup or a fraud attempt; the tells show up in what the system computes, not in
        what you type. Nothing here is generated for you.
      </p>

      {!result ? (
        <form onSubmit={submit}>
          <div className="grid grid-2" style={{ alignItems: "start" }}>
            <div className="card">
              <div className="section-heading">1. Personal details</div>
              {[
                ["name", "Full name", "text"],
                ["date_of_birth", "Date of birth", "date"],
                ["address", "Address", "text"],
                ["phone", "Phone", "text"],
                ["email", "Email", "text"],
              ].map(([key, label, type]) => (
                <div className="field" key={key}>
                  <label>{label}</label>
                  <input
                    type={type}
                    required
                    value={form[key]}
                    onChange={set(key)}
                    onFocus={onFieldFocus}
                    onBlur={onFieldBlur}
                  />
                </div>
              ))}
              <div className="field">
                <label>Document type</label>
                <select value={form.document_type} onChange={set("document_type")} onFocus={onFieldFocus} onBlur={onFieldBlur}>
                  <option value="NATIONAL_ID">National ID</option>
                  <option value="PASSPORT">Passport</option>
                  <option value="DRIVERS_LICENSE">Driver's License</option>
                </select>
              </div>
              <div className="field">
                <label>ID number (as shown on document, optional)</label>
                <input type="text" value={form.document_number} onChange={set("document_number")} onFocus={onFieldFocus} onBlur={onFieldBlur} />
              </div>
              {fingerprint && (
                <p style={{ fontSize: 11, color: "var(--text-dim)" }}>
                  Device fingerprint captured automatically: {fingerprint.os} · {fingerprint.browser} · first seen{" "}
                  {fingerprint.device_age_days === 0 ? "today" : `${fingerprint.device_age_days}d ago`} on this browser.
                </p>
              )}
            </div>

            <div>
              <div className="card" style={{ marginBottom: 16 }}>
                <div className="section-heading">2. Document photo</div>
                <div className="pill-row" style={{ marginBottom: 12 }}>
                  <a className="pill" href={SampleDocumentUrl()} target="_blank" rel="noreferrer">
                    ⬇ Download clean sample ID
                  </a>
                  <a className="pill" href={SampleDocumentUrl({ blur: true, noise: true, rotate: 4, tamper_fields: true })} target="_blank" rel="noreferrer">
                    ⬇ Download tampered sample ID
                  </a>
                </div>
                <p style={{ fontSize: 11.5, color: "var(--text-dim)", marginBottom: 10 }}>
                  Don't have a real ID handy for testing? Download one of the clearly-labeled synthetic sample
                  documents above, then upload it below. Type a name that matches (or doesn't match) the sample to
                  see the consistency check react.
                </p>
                <input type="file" accept="image/*" onChange={onDocumentChange} />
                {documentPreview && (
                  <img src={documentPreview} alt="document preview" style={{ marginTop: 10, width: "100%", maxWidth: 320, borderRadius: 8, border: "1px solid var(--panel-border)" }} />
                )}
              </div>

              <div className="card">
                <div className="section-heading">3. Live selfie</div>
                <p style={{ fontSize: 11.5, color: "var(--text-dim)", marginBottom: 10 }}>
                  Captured locally in your browser and sent only to this app's own backend for scoring -- never stored
                  or sent anywhere else.
                </p>
                <WebcamCapture onCapture={setSelfieBlob} />
              </div>
            </div>
          </div>

          <div style={{ marginTop: 20 }}>
            <button className="btn" type="submit" disabled={submitting}>
              {submitting ? "Submitting…" : "Submit Application"}
            </button>
          </div>
        </form>
      ) : (
        <ResultView result={result} onReset={reset} />
      )}

      <Toast message={error} type="error" onClose={() => setError(null)} />
    </div>
  );
}

function ResultView({ result, onReset }) {
  const { applicant, verification, notes } = result;
  return (
    <div>
      {verification ? (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="section-heading">Verification Result</div>
          <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 14 }}>
            <div style={{ fontSize: 34, fontWeight: 800 }}>{verification.risk_score}% RISK</div>
            <DecisionBadge decision={verification.decision} />
          </div>
          {verification.risk_factors.length === 0 ? (
            <div className="empty-state">No elevated risk factors -- looks clean.</div>
          ) : (
            verification.risk_factors.map((f, i) => (
              <div className="risk-factor" key={i}>
                <span>{i + 1}. {f.description}</span>
              </div>
            ))
          )}
          <p style={{ color: "var(--text-dim)", fontSize: 12, marginTop: 10 }}>Model v{verification.model_version} · applicant_id {applicant.applicant_id}</p>
        </div>
      ) : (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="empty-state">Submission saved, but no trained model yet -- run Blue Team → RUN BLUE TEAM first, then re-score this applicant from the Blue Team page.</div>
        </div>
      )}

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-heading">What the system actually computed</div>
        <div className="grid grid-3">
          <MiniField label="Device reuse" value={applicant.device_reuse_count} />
          <MiniField label="IP reuse" value={applicant.ip_reuse_count} />
          <MiniField label="Name match" value={applicant.name_match_score} />
          <MiniField label="DOB match" value={applicant.dob_match_score} />
          <MiniField label="Doc tamper score" value={applicant.document_tamper_score} />
          <MiniField label="OCR confidence" value={applicant.ocr_confidence} />
          <MiniField label="Face similarity" value={applicant.face_similarity_score} />
          <MiniField label="Liveness" value={applicant.liveness_score} />
          <MiniField label="Automation score" value={applicant.automation_score} />
        </div>
      </div>

      {notes?.length > 0 && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="section-heading" style={{ fontSize: 13 }}>Pipeline notes</div>
          {notes.map((n, i) => (
            <p key={i} style={{ fontSize: 12, color: "var(--text-dim)", marginBottom: 6 }}>· {n}</p>
          ))}
        </div>
      )}

      <button className="btn secondary" onClick={onReset}>Submit another application</button>
    </div>
  );
}

function MiniField({ label, value }) {
  return (
    <div>
      <div style={{ fontSize: 10.5, color: "var(--text-dim)", textTransform: "uppercase", fontWeight: 700 }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 800 }}>{value ?? "–"}</div>
    </div>
  );
}

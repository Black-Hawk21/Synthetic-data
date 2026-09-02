import axios from "axios";

const api = axios.create({ baseURL: "/api", timeout: 120000 });
const root = axios.create({ baseURL: "/", timeout: 15000 });

// Every function here calls the REAL FastAPI backend -- no mocked data
// anywhere in this file (section 32).
export const Health = () => root.get("/health").then((r) => r.data);

export const GenerateApplicants = (n, seed) =>
  api.post("/generate-applicants", { n, seed }).then((r) => r.data);

export const DatasetSummary = () => api.get("/dataset-summary").then((r) => r.data);

export const ListAttacks = () => api.get("/attacks").then((r) => r.data);

export const GenerateAttack = (attack_type, difficulty, n) =>
  api.post("/generate-attack", { attack_type, difficulty, n }).then((r) => r.data);

export const RunRedTeam = (attacks) =>
  api.post("/run-red-team", { attacks }).then((r) => r.data);

export const RunBlueTeam = (seed = 42) =>
  api.post("/run-blue-team", { seed }).then((r) => r.data);

export const ScoreApplicant = (applicant_id, top_n_factors = 5) =>
  api.post("/score-applicant", { applicant_id, top_n_factors }).then((r) => r.data);

export const RunDetection = (threshold) =>
  api.post("/run-detection", { threshold }).then((r) => r.data);

export const AttackResults = () => api.get("/attack-results").then((r) => r.data);

export const ModelInfo = () => api.get("/model-info").then((r) => r.data);

export const Metrics = () => api.get("/metrics").then((r) => r.data);

export const GraphForApplicant = (applicantId) =>
  api.get(`/graph/${applicantId}`).then((r) => r.data);

export const GraphRings = (minSize = 3) =>
  api.get("/graph/rings", { params: { min_size: minSize } }).then((r) => r.data);

export const RunClosedLoop = (iterations, n_per_type, use_llm, seed = 42) =>
  api.post("/run-closed-loop", { iterations, n_per_type, use_llm, seed }).then((r) => r.data);

export const FeedbackHistory = () => api.get("/feedback").then((r) => r.data);

export const SimulateApplicant = (attack_type, difficulty, seed) =>
  api.post("/onboarding/simulate", { attack_type, difficulty, seed }).then((r) => r.data);

export const SampleDocumentUrl = (opts = {}) => {
  const params = new URLSearchParams(opts).toString();
  return `/api/document/sample${params ? `?${params}` : ""}`;
};

// Real manual KYC submission: multipart/form-data with typed fields, an
// uploaded document image, a captured selfie blob, and real client-side
// telemetry/device-fingerprint JSON blobs.
export const SubmitManualOnboarding = (fields, documentFile, selfieBlob, telemetry, deviceFingerprint) => {
  const form = new FormData();
  Object.entries(fields).forEach(([k, v]) => form.append(k, v ?? ""));
  form.append("telemetry", JSON.stringify(telemetry));
  form.append("device_fingerprint", JSON.stringify(deviceFingerprint));
  if (documentFile) form.append("document_image", documentFile, documentFile.name || "document.png");
  if (selfieBlob) form.append("selfie_image", selfieBlob, "selfie.jpg");
  return api.post("/onboarding/submit", form).then((r) => r.data);
};

export default api;

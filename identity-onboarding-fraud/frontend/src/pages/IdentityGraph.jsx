import { useCallback, useEffect, useMemo, useState } from "react";
import ReactFlow, { Background, Controls, MarkerType } from "reactflow";
import "reactflow/dist/style.css";
import Toast from "../components/Toast.jsx";
import { GraphForApplicant, GraphRings } from "../services/api.js";

const KIND_COLOR = {
  person: "#4f8cff",
  phone: "#35d08f",
  email: "#ffb454",
  address: "#ff8a5a",
  device: "#7c5cff",
  ip: "#ff5470",
  document: "#8b97ac",
};

function buildElements(applicantId, info) {
  const nodes = [
    {
      id: `center:${applicantId}`,
      position: { x: 0, y: 0 },
      data: { label: `👤 ${applicantId.slice(-8)}` },
      style: nodeStyle(KIND_COLOR.person, true),
    },
  ];
  const edges = [];
  const angleStep = (2 * Math.PI) / Math.max(info.shared_attributes.length, 1);

  info.shared_attributes.forEach((attr, i) => {
    const attrId = `attr:${attr.attribute_type}:${attr.value}`;
    const r = 260;
    const angle = i * angleStep;
    nodes.push({
      id: attrId,
      position: { x: Math.cos(angle) * r, y: Math.sin(angle) * r },
      data: { label: `${attr.attribute_type}\n${String(attr.value).slice(0, 16)}` },
      style: nodeStyle(KIND_COLOR[attr.attribute_type] || "#8b97ac", false),
    });
    edges.push({
      id: `e:${applicantId}:${attrId}`,
      source: `center:${applicantId}`,
      target: attrId,
      style: { stroke: "#2c3850" },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#2c3850" },
    });

    const coApplicants = (attr.shared_with || []).slice(0, 4);
    coApplicants.forEach((coId, j) => {
      const coNodeId = `co:${attrId}:${coId}`;
      const subAngle = angle + (j - coApplicants.length / 2) * 0.28;
      const subR = r + 170;
      nodes.push({
        id: coNodeId,
        position: { x: Math.cos(subAngle) * subR, y: Math.sin(subAngle) * subR },
        data: { label: `👤 ${coId.slice(-8)}` },
        style: nodeStyle("#3a4560", false, true),
      });
      edges.push({
        id: `e:${attrId}:${coNodeId}`,
        source: attrId,
        target: coNodeId,
        style: { stroke: "#232c3d" },
      });
    });
  });

  return { nodes, edges };
}

function nodeStyle(color, isCenter, dim = false) {
  return {
    background: dim ? "#161d2b" : "#131a26",
    color: dim ? "#8b97ac" : "#e7ecf4",
    border: `2px solid ${color}`,
    borderRadius: isCenter ? 999 : 10,
    padding: 8,
    fontSize: 10.5,
    fontWeight: 700,
    whiteSpace: "pre-line",
    textAlign: "center",
    width: isCenter ? 130 : 110,
  };
}

export default function IdentityGraph() {
  const [applicantId, setApplicantId] = useState("");
  const [info, setInfo] = useState(null);
  const [rings, setRings] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    GraphRings(3).then(setRings).catch(() => {});
  }, []);

  const load = useCallback(async (id) => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const data = await GraphForApplicant(id);
      setInfo(data);
      setApplicantId(id);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const { nodes, edges } = useMemo(() => (info ? buildElements(applicantId, info) : { nodes: [], edges: [] }), [info, applicantId]);

  return (
    <div>
      <h1 className="page-title">Identity Graph</h1>
      <p className="page-subtitle">Shared devices, IPs, phones, emails and addresses -- the structure a single-field rule can't see.</p>

      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: "flex", gap: 10 }}>
          <input
            type="text"
            placeholder="Enter applicant_id (e.g. APP_xxxxxxxxxxxxxxxx)"
            value={applicantId}
            onChange={(e) => setApplicantId(e.target.value)}
            style={{ flex: 1 }}
          />
          <button className="btn" onClick={() => load(applicantId)} disabled={loading}>
            {loading ? "Loading…" : "Load Graph"}
          </button>
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "1fr 320px", gap: 20, alignItems: "start" }}>
        <div className="card" style={{ height: 560, padding: 0, overflow: "hidden" }}>
          {nodes.length === 0 ? (
            <div className="empty-state" style={{ paddingTop: 240 }}>
              Load an applicant, or click one from a fraud ring on the right, to visualize their identity graph.
            </div>
          ) : (
            <ReactFlow nodes={nodes} edges={edges} fitView proOptions={{ hideAttribution: true }}>
              <Background color="#1a2233" gap={20} />
              <Controls />
            </ReactFlow>
          )}
        </div>

        <div>
          {info && (
            <div className="card" style={{ marginBottom: 16 }}>
              <div className="section-heading">Applicant summary</div>
              <div className="risk-factor"><span>Risk score</span><span>{info.risk_score?.toFixed(1)}%</span></div>
              <div className="risk-factor"><span>Ground truth</span><span>{info.is_fraud ? info.attack_type : "legitimate"}</span></div>
              <div className="risk-factor"><span>Connected identities</span><span>{Math.max(info.connected_component_size - 1, 0)}</span></div>
              <div className="risk-factor"><span>Shared attributes</span><span>{info.shared_attributes.length}</span></div>
            </div>
          )}

          <div className="card">
            <div className="section-heading">Suspicious clusters</div>
            {!rings || rings.clusters.length === 0 ? (
              <div className="empty-state">None detected yet.</div>
            ) : (
              rings.clusters.slice(0, 8).map((c) => (
                <div
                  key={c.cluster_id}
                  className="risk-factor"
                  style={{ cursor: "pointer" }}
                  onClick={() => load(c.applicant_ids[0])}
                >
                  <span>
                    {c.size} identities · score {c.avg_suspicious_score.toFixed(2)}
                    {c.fraud_rate != null && ` · ${(c.fraud_rate * 100).toFixed(0)}% fraud`}
                  </span>
                  <span style={{ color: "var(--accent)" }}>view →</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <Toast message={error} type="error" onClose={() => setError(null)} />
    </div>
  );
}

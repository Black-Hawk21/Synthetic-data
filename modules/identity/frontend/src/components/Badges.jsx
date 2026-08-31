export function DecisionBadge({ decision }) {
  if (!decision) return null;
  return <span className={`badge ${decision.toLowerCase()}`}>{decision}</span>;
}

export function SeverityBadge({ severity }) {
  if (!severity) return null;
  return <span className={`badge ${severity.toLowerCase()}`}>{severity}</span>;
}

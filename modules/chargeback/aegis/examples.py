"""Loads the curated, pre-generated fraud example gallery (data/fraud_samples.json)
for the UI's Example Gallery tab -- a fully offline fallback that renders real,
previously-generated Red Team vs Blue Team rounds without depending on a live
network/API call working at demo time. Regenerate with scripts/generate_fraud_samples.py.
"""

import json
from pathlib import Path

from aegis.schemas import RoundRecord

EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "fraud_samples.json"


def load_example_gallery() -> list[list[RoundRecord]]:
    """Each entry is one example's full round sequence (usually length 1, occasionally
    longer for the mutation-escalation showcase)."""
    if not EXAMPLES_PATH.exists():
        return []
    raw = json.loads(EXAMPLES_PATH.read_text())
    return [[RoundRecord(**record) for record in sequence] for sequence in raw]

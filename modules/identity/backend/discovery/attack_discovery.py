"""Attack Discovery Agent (section 6): analyzes detector weaknesses and
proposes new attack hypotheses. Tries the local LLM first (if running),
falls back to the deterministic rule-based engine, which ALWAYS works."""
from __future__ import annotations

import logging

from backend.discovery.fallback import propose_hypotheses as fallback_propose
from backend.discovery.local_llm import generate_attack_hypotheses_via_llm
from backend.red_team.registry import list_attacks

logger = logging.getLogger(__name__)


def discover_attack_hypotheses(per_attack_type_recall: dict, n: int = 5, use_llm: bool = True) -> dict:
    llm_result = None
    if use_llm:
        llm_result = generate_attack_hypotheses_via_llm(per_attack_type_recall, list_attacks(), n=n)

    if llm_result:
        return {"source": "ollama_llm", "hypotheses": llm_result}

    return {"source": "rule_based_fallback", "hypotheses": fallback_propose(per_attack_type_recall, n=n)}

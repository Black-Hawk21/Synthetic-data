"""Strategy-pattern base class for all red-team attack generators.

Every concrete attack (backend/red_team/*.py) subclasses AttackStrategy and
implements `mutate()`. Nothing outside this module needs to know how many
attack types exist or how they work internally -- see registry.py.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from backend.data.generator import generate_legitimate_applicants


def severity_from_difficulty(difficulty: float) -> str:
    """Counter-intuitive but correct: a HARDER (higher-difficulty) attack to
    detect is the MORE severe one from a risk standpoint, because it is more
    likely to slip past the detector."""
    if difficulty >= 0.8:
        return "CRITICAL"
    if difficulty >= 0.55:
        return "HIGH"
    if difficulty >= 0.3:
        return "MEDIUM"
    return "LOW"


class AttackStrategy(ABC):
    attack_type: str = "BASE"
    features_affected: list[str] = []
    summary: str = "Base attack strategy."

    def description(self) -> str:
        return self.summary

    def severity(self, difficulty: float) -> str:
        return severity_from_difficulty(difficulty)

    @abstractmethod
    def mutate(self, df: pd.DataFrame, rng: np.random.Generator, difficulty: float) -> pd.DataFrame:
        """Mutate a batch of otherwise-legitimate-looking applicants in
        place (on a copy) to encode this attack's fraud pattern. `difficulty`
        in [0, 1]: 0 = blatant/easy-to-catch, 1 = highly subtle/near-legit.
        Must return the mutated dataframe."""
        raise NotImplementedError

    def generate(self, n: int, difficulty: float = 0.5, seed: int | None = None) -> pd.DataFrame:
        """Generate `n` synthetic fraud records for this attack type."""
        rng = np.random.default_rng(seed)
        base_seed = int(rng.integers(0, 2_000_000_000))
        base = generate_legitimate_applicants(n, seed=base_seed)
        mutated = self.mutate(base, rng, float(np.clip(difficulty, 0.0, 1.0)))
        mutated["is_fraud"] = 1
        mutated["attack_type"] = self.attack_type
        mutated["difficulty"] = float(np.clip(difficulty, 0.0, 1.0))
        mutated["attack_id"] = [f"ATK_{self.attack_type}_{uuid.uuid4().hex[:10]}" for _ in range(len(mutated))]
        mutated["source"] = "synthetic_attack"
        return mutated

    def to_meta(self, difficulty: float = 0.5) -> dict:
        return {
            "attack_type": self.attack_type,
            "description": self.description(),
            "severity": self.severity(difficulty),
            "features_affected": self.features_affected,
        }

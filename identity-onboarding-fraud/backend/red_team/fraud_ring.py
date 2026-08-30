from __future__ import annotations

import numpy as np

from backend.red_team.base import AttackStrategy
from backend.red_team.registry import register_attack
from backend.red_team.utils import blend, shared_pool, tight_timestamps


@register_attack
class FraudRingAttack(AttackStrategy):
    """A coordinated ring of otherwise-unrelated-looking identities (different
    names, DOBs, faces) that all share underlying infrastructure -- the
    canonical multi-node fraud ring the identity graph is built to expose."""

    attack_type = "FRAUD_RING"
    features_affected = [
        "device_reuse_count", "ip_reuse_count", "connected_component_size",
        "suspicious_cluster_score",
    ]
    summary = "A ring of apparently-unrelated identities shares devices, IPs and/or addresses -- exposed by identity-graph clustering rather than any single field."

    def mutate(self, df, rng, difficulty):
        n = len(df)
        # Rings of 4-15 identities, tighter/smaller and infra shared more
        # selectively as difficulty increases.
        lo = int(round(blend(6, 3, difficulty)))
        hi = int(round(blend(15, 5, difficulty)))
        hi = max(hi, lo + 1)
        device_pool = shared_pool(rng, n, (lo, hi), "ring_dev")
        ip_pool = shared_pool(rng, n, (lo, hi), "ring_ip")
        share_frac_device = blend(0.9, 0.5, difficulty)
        share_frac_ip = blend(0.85, 0.45, difficulty)
        mask_d = rng.random(n) < share_frac_device
        mask_i = rng.random(n) < share_frac_ip
        df.loc[mask_d, "device_id"] = device_pool[mask_d]
        df.loc[mask_i, "ip_id"] = ip_pool[mask_i]
        return df


@register_attack
class CoordinatedOnboardingAttack(AttackStrategy):
    """A single operator (or small team) runs many onboarding sessions back
    to back with shared infrastructure AND a tight submission cadence --
    fraud ring + velocity combined."""

    attack_type = "COORDINATED_ONBOARDING"
    features_affected = [
        "device_reuse_count", "ip_reuse_count", "application_velocity",
        "automation_score",
    ]
    summary = "Multiple applications from a shared operator: common infrastructure plus a tight, semi-automated submission cadence."

    def mutate(self, df, rng, difficulty):
        n = len(df)
        lo = int(round(blend(5, 3, difficulty)))
        hi = int(round(blend(12, 5, difficulty)))
        hi = max(hi, lo + 1)
        device_pool = shared_pool(rng, n, (lo, hi), "coord_dev")
        share_frac = blend(0.8, 0.4, difficulty)
        mask = rng.random(n) < share_frac
        df.loc[mask, "device_id"] = device_pool[mask]
        window_minutes = blend(30, 480, difficulty)
        df["created_at"] = tight_timestamps(rng, n, window_minutes)
        df["application_velocity"] = np.clip(rng.normal(blend(3.0, 0.9, difficulty), 0.6, size=n), 0.05, None)
        df["automation_score"] = np.clip(df["automation_score"] + blend(0.35, 0.08, difficulty), 0, 1)
        return df

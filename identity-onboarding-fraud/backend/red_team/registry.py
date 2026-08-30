"""Plugin registry for attack strategies. New attacks self-register with
`@register_attack` and are immediately available to the API, CLI scripts,
and the discovery/feedback engines -- no other file needs to change."""
from __future__ import annotations

from backend.red_team.base import AttackStrategy

_REGISTRY: dict[str, AttackStrategy] = {}


def register_attack(cls):
    instance = cls()
    if not getattr(instance, "attack_type", None):
        raise ValueError(f"{cls.__name__} must define attack_type")
    _REGISTRY[instance.attack_type] = instance
    return cls


def get_attack(attack_type: str) -> AttackStrategy:
    if attack_type not in _REGISTRY:
        raise KeyError(f"Unknown attack_type '{attack_type}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[attack_type]


def list_attacks() -> list[str]:
    return sorted(_REGISTRY.keys())


def all_attack_meta(difficulty: float = 0.5) -> list[dict]:
    return [a.to_meta(difficulty) for a in _REGISTRY.values()]


def load_all() -> None:
    """Import every attack module so its @register_attack decorator runs."""
    from backend.red_team import (  # noqa: F401
        synthetic_identity, identity_reuse, document_tampering,
        face_mismatch, liveness_spoof, device_reuse, velocity, fraud_ring,
    )

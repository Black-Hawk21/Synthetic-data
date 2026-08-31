"""Single-attribute infrastructure-reuse attacks: DEVICE_REUSE, IP_REUSE,
PHONE_REUSE, EMAIL_REUSE, ADDRESS_REUSE.

Kept in one file since they share an identical mechanic (force a cluster of
rows onto one shared attribute value) and differ only in which column and
target reuse magnitude they touch.
"""
from __future__ import annotations

import numpy as np

from backend.red_team.base import AttackStrategy
from backend.red_team.registry import register_attack
from backend.red_team.utils import blend, shared_pool


class _ReuseAttackBase(AttackStrategy):
    column: str = ""
    prefix: str = "reuse"
    easy_cluster: tuple[int, int] = (8, 25)
    hard_cluster: tuple[int, int] = (2, 4)
    share_easy: float = 0.95
    share_hard: float = 0.30

    def mutate(self, df, rng, difficulty):
        n = len(df)
        lo = int(round(blend(self.easy_cluster[0], self.hard_cluster[0], difficulty)))
        hi = int(round(blend(self.easy_cluster[1], self.hard_cluster[1], difficulty)))
        hi = max(hi, lo + 1)
        pool = shared_pool(rng, n, (lo, hi), self.prefix)
        share_frac = blend(self.share_easy, self.share_hard, difficulty)
        mask = rng.random(n) < share_frac
        df.loc[mask, self.column] = pool[mask]
        return df


@register_attack
class DeviceReuseAttack(_ReuseAttackBase):
    attack_type = "DEVICE_REUSE"
    column = "device_id"
    prefix = "atk_dev"
    features_affected = ["device_reuse_count", "device_identity_count"]
    summary = "Many distinct-looking synthetic identities are onboarded from the same physical/emulated device."


@register_attack
class IPReuseAttack(_ReuseAttackBase):
    attack_type = "IP_REUSE"
    column = "ip_id"
    prefix = "atk_ip"
    features_affected = ["ip_reuse_count", "identities_from_ip", "applications_from_ip"]
    summary = "Many distinct-looking synthetic identities are onboarded from the same IP address (proxy farm / botnet exit)."


@register_attack
class PhoneReuseAttack(_ReuseAttackBase):
    attack_type = "PHONE_REUSE"
    column = "phone"
    prefix = "atk_ph"
    features_affected = ["phone_reuse_count", "identity_reuse_count"]
    summary = "The same phone number (VOIP pool / SIM farm) is registered against many synthetic identities."


@register_attack
class EmailReuseAttack(_ReuseAttackBase):
    attack_type = "EMAIL_REUSE"
    column = "email"
    prefix = "atk_em"
    features_affected = ["email_reuse_count"]
    summary = "The same email inbox (or a disposable-email pattern) is reused to register many synthetic identities."


@register_attack
class AddressReuseAttack(_ReuseAttackBase):
    attack_type = "ADDRESS_REUSE"
    column = "address"
    prefix = "atk_addr"
    features_affected = ["address_reuse_count", "shared_address_count"]
    summary = "The same mailing/drop address is used to register many synthetic identities (mail-drop fraud)."

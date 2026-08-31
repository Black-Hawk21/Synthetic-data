"""Account archetypes and their behavioural parameter ranges.

Each archetype is a *range* per parameter; every account samples its own point
from that range, so no two salary accounts behave identically. This matters:
if all normal accounts share one distribution, a detector can separate
laundering with a single threshold and your dataset teaches nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

CITIES = [
    "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune",
    "Kolkata", "Ahmedabad", "Jaipur", "Kochi",
]
FOREIGN_COUNTRIES = ["AE", "SG", "GB", "US", "HK", "MY", "MU", "CY"]


@dataclass(frozen=True)
class Archetype:
    name: str
    is_business: bool
    out_per_day: Tuple[float, float]      # outgoing transactions per day
    amount_median: Tuple[float, float]    # INR
    amount_sigma: Tuple[float, float]     # log-sigma of the amount distribution
    night_ratio: Tuple[float, float]      # share of activity between 23:00-06:00
    popularity: float                     # attractiveness as a *receiver*
    monthly_income: Tuple[float, float]
    business_hours: bool = False
    salaried: bool = False
    employs: bool = False
    tags: List[str] = field(default_factory=list)


ARCHETYPES: Dict[str, Archetype] = {
    "salary": Archetype(
        "salary", False, (0.15, 0.80), (600, 3500), (0.85, 1.25), (0.02, 0.10),
        popularity=1.0, monthly_income=(35_000, 250_000), salaried=True,
        tags=["retail"]),
    "student": Archetype(
        "student", False, (0.08, 0.50), (120, 900), (0.75, 1.10), (0.06, 0.22),
        popularity=0.7, monthly_income=(4_000, 25_000), tags=["retail", "low_value"]),
    "freelancer": Archetype(
        "freelancer", False, (0.10, 0.60), (500, 6_000), (1.05, 1.60), (0.05, 0.18),
        popularity=2.5, monthly_income=(20_000, 400_000), tags=["retail", "irregular"]),
    "household": Archetype(
        "household", False, (0.15, 0.70), (400, 4_000), (0.80, 1.20), (0.02, 0.09),
        popularity=1.4, monthly_income=(25_000, 200_000), tags=["retail"]),
    "small_business": Archetype(
        "small_business", True, (0.50, 3.00), (1_500, 25_000), (0.95, 1.45), (0.02, 0.08),
        popularity=6.0, monthly_income=(200_000, 4_000_000), business_hours=True,
        employs=True, tags=["business"]),
    "merchant": Archetype(
        "merchant", True, (1.00, 6.00), (250, 4_000), (0.70, 1.10), (0.03, 0.14),
        popularity=45.0, monthly_income=(300_000, 9_000_000), business_hours=True,
        employs=True, tags=["business", "high_in_degree"]),
    "large_business": Archetype(
        "large_business", True, (2.00, 12.00), (25_000, 400_000), (1.10, 1.70), (0.01, 0.06),
        popularity=18.0, monthly_income=(5_000_000, 250_000_000), business_hours=True,
        employs=True, tags=["business", "high_value"]),
    "investment": Archetype(
        "investment", True, (0.05, 0.40), (150_000, 2_500_000), (1.00, 1.55), (0.01, 0.05),
        popularity=9.0, monthly_income=(1_000_000, 80_000_000), business_hours=True,
        tags=["business", "high_value", "low_frequency"]),
}

# Roles used inside episodes (both laundering and benign).
ROLES = [
    "source", "intermediary", "mule", "collector", "destination",
    "transit", "beneficiary", "counterparty",
]

# Which archetypes make plausible participants for each structural role.
SOURCE_POOL = ["large_business", "investment", "small_business", "merchant"]
MULE_POOL = ["student", "freelancer", "salary", "household"]
SHELL_POOL = ["small_business", "freelancer", "investment"]

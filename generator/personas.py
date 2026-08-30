"""
Synthetic victim persona generator, built on Faker (en_IN locale) so names,
cities, and other details are genuinely varied instead of drawn from a
short hardcoded list -- the earlier hand-rolled version only had 16 first
names x 15 last names (240 combos), which was a real contributor to the
duplicate/near-duplicate outputs seen in the first generation run.

All personas are fabricated (Faker generates fictional names, not real
people). Used to condition attack generation so messages feel
targeted/personalized, mirroring how a real GenAI-powered fraud pipeline
would use scraped or leaked context.

Requires: pip install faker --break-system-packages
"""

import random
from faker import Faker

_faker_en_in = Faker("en_IN")

BANKS = [
    "HDFC Bank", "ICICI Bank", "Axis Bank", "State Bank of India",
    "Kotak Mahindra Bank", "Yes Bank", "IndusInd Bank", "Punjab National Bank",
    "Bank of Baroda", "IDFC FIRST Bank",
]

PAYMENT_APPS = ["Paytm", "PhonePe", "Google Pay", "Mastercard SecureCode", "Amazon Pay", "BHIM UPI"]

INCOME_BRACKETS = ["student/low-income", "salaried-mid", "salaried-high", "retired/pension", "small-business-owner"]

RECENT_TRANSACTION_TYPES = [
    "online shopping purchase",
    "utility bill payment",
    "international wire transfer",
    "credit card EMI payment",
    "subscription renewal",
    "peer-to-peer transfer to a friend",
    "ATM cash withdrawal",
    "mobile recharge",
    "insurance premium payment",
    "rent payment via UPI",
]

GULLIBILITY_LEVELS = ["low", "medium", "high"]  # used for conversational victim-sim


def generate_persona(rng: random.Random = None, seed: int = None) -> dict:
    """rng controls the choice() calls for the curated lists (bank, app, etc);
    seed (if given) also seeds Faker for this specific persona so repeated
    calls with the same seed reproduce the same name/city, which is handy
    for debugging a specific generated sample."""
    rng = rng or random
    if seed is not None:
        Faker.seed(seed)
    return {
        "name": _faker_en_in.name(),
        "city": _faker_en_in.city(),
        "job": _faker_en_in.job(),
        "bank": rng.choice(BANKS),
        "payment_app": rng.choice(PAYMENT_APPS),
        "income_bracket": rng.choice(INCOME_BRACKETS),
        "recent_transaction": rng.choice(RECENT_TRANSACTION_TYPES),
        "gullibility": rng.choice(GULLIBILITY_LEVELS),
    }


def generate_personas(n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    Faker.seed(seed)
    # NOTE: we don't reseed Faker per-persona here (that would make every
    # persona in the batch derive from the same rng draw order and could
    # cycle back to repeats for large n) -- Faker's own internal generator
    # advances across calls, so a single seed() at the top keeps the whole
    # batch reproducible while still giving n distinct-ish identities.
    return [generate_persona(rng) for _ in range(n)]


if __name__ == "__main__":
    for p in generate_personas(8):
        print(p)

"""
tests/synthetic_data.py

Synthetic battle histories for demoing the learning loop without a real
battle engine wired up yet. Three opponent personalities:

  A. Aggressive    - attacks almost always, ignores being disadvantaged,
                      and reliably attacks right after switching in
                      (a deliberate pattern for the learner to pick up).
  B. Defensive     - switches/defends when disadvantaged, rarely
                      attacks while behind.
  C. Unpredictable - roughly uniform random, no strong pattern.
"""

import random
from prediction_engine.core.interfaces import Action

ATTACK = Action(name="attack", category="attack", power=0.6)
SWITCH = Action(name="switch", category="switch")
DEFEND = Action(name="defend", category="defend")

ACTIONS = [ATTACK, SWITCH, DEFEND]
_BY_NAME = {"attack": ATTACK, "switch": SWITCH, "defend": DEFEND}

PERSONALITIES = ["aggressive", "defensive", "unpredictable"]


def _pick(rng: random.Random, weights: dict) -> Action:
    names = list(weights.keys())
    probs = list(weights.values())
    choice = rng.choices(names, weights=probs, k=1)[0]
    return _BY_NAME[choice]


def generate_history(personality: str, n_turns: int = 40, seed: int = 0):
    """
    Returns a list of dicts: {action, was_disadvantaged, took_damage},
    one per turn, generated according to a personality profile. Meant
    to be fed turn-by-turn into an OpponentLearner.
    """
    if personality not in PERSONALITIES:
        raise ValueError(f"unknown personality: {personality}")

    rng = random.Random(seed)
    turns = []
    prev_action_name = None

    for _ in range(n_turns):
        was_disadvantaged = rng.random() < 0.3
        took_damage = rng.random() < 0.4

        if personality == "aggressive":
            if prev_action_name == "switch":
                weights = {"attack": 0.9, "switch": 0.05, "defend": 0.05}  # signature pattern
            elif was_disadvantaged:
                weights = {"attack": 0.75, "switch": 0.15, "defend": 0.10}
            else:
                weights = {"attack": 0.8, "switch": 0.1, "defend": 0.1}

        elif personality == "defensive":
            if was_disadvantaged:
                weights = {"attack": 0.15, "switch": 0.5, "defend": 0.35}
            elif took_damage:
                weights = {"attack": 0.3, "switch": 0.3, "defend": 0.4}
            else:
                weights = {"attack": 0.4, "switch": 0.2, "defend": 0.4}

        else:  # unpredictable
            weights = {"attack": 0.34, "switch": 0.33, "defend": 0.33}

        action = _pick(rng, weights)
        turns.append({
            "action": action,
            "was_disadvantaged": was_disadvantaged,
            "took_damage": took_damage,
        })
        prev_action_name = action.name

    return turns

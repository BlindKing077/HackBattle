"""
core/predictor.py

Next-action prediction. Blends four signals, each weighted by how much
evidence backs it (a simple Bayesian-model-averaging-style approach):

  1. transitions_by_prev[prev_action]     - "what do they do after X"
  2. transitions_by_disadvantage[0/1]     - reaction to being disadvantaged
  3. transitions_by_damage[0/1]           - reaction to taking damage
  4. general behavioral tendencies        - always-available fallback prior

Each conditional signal only pulls weight proportional to its own
sample count, so a pattern with little evidence can't drown out the
others, and the overall prediction confidence is explicitly capped by
total observations - this MVP never claims certainty it hasn't earned.
"""

from __future__ import annotations
from typing import List, Optional, Dict
from .interfaces import GameState, Action, OpponentModel, Prediction

# how many total observations before overall confidence is unclamped
MIN_SAMPLES_FOR_FULL_CONFIDENCE = 8
# pseudo-sample-count weight given to the general tendency prior; keeps
# it influential until conditional tables have accumulated real evidence
PRIOR_PSEUDO_COUNT = 4
# cap any single conditional table's influence so one noisy bucket can't
# swamp everything else
MAX_BUCKET_WEIGHT = 12


def _tendency_prior(action: Action, tendencies: dict) -> float:
    """Rough prior weight for an action from general tendencies alone."""
    if action.category == "attack":
        return 0.3 + 0.6 * tendencies.get("aggression", 0.5)
    if action.category == "switch":
        return 0.3 + 0.6 * tendencies.get("switch_tendency", 0.5)
    if action.category == "defend":
        return 0.3 + 0.6 * tendencies.get("defensive_tendency", 0.5)
    return 0.3


def _bucket_distribution(bucket: Optional[Dict[str, int]], action_names: List[str]):
    """Normalize a {action_name: count} bucket into a distribution + its sample size."""
    if not bucket:
        return {}, 0
    total = sum(bucket.get(name, 0) for name in action_names)
    if total == 0:
        return {}, 0
    return {name: bucket.get(name, 0) / total for name in action_names}, total


def predict_next_action(
    state: GameState,
    model: OpponentModel,
    available_actions: List[Action],
    prev_action: Optional[Action],
    was_disadvantaged: bool,
    took_damage: bool,
) -> Prediction:
    if not available_actions:
        return Prediction(predicted_action="unknown", confidence=0.0, alternatives=[])

    names = [a.name for a in available_actions]

    prior_raw = {a.name: _tendency_prior(a, model.tendencies) for a in available_actions}
    prior_total = sum(prior_raw.values()) or 1.0
    prior_dist = {name: v / prior_total for name, v in prior_raw.items()}

    prev_key = prev_action.name if prev_action else "none"
    prev_dist, prev_n = _bucket_distribution(model.transitions_by_prev.get(prev_key), names)
    disadv_dist, disadv_n = _bucket_distribution(
        model.transitions_by_disadvantage.get(str(int(was_disadvantaged))), names
    )
    damage_dist, damage_n = _bucket_distribution(
        model.transitions_by_damage.get(str(int(took_damage))), names
    )

    # weighted (sample-size-proportional) blend across all available signals
    weights_total = float(PRIOR_PSEUDO_COUNT)
    combined = {name: PRIOR_PSEUDO_COUNT * prior_dist[name] for name in names}
    for dist, n in ((prev_dist, prev_n), (disadv_dist, disadv_n), (damage_dist, damage_n)):
        w = min(n, MAX_BUCKET_WEIGHT)
        if w > 0:
            weights_total += w
            for name in names:
                combined[name] += w * dist.get(name, 0.0)

    for name in names:
        combined[name] /= weights_total

    ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)
    top_action, top_conf = ranked[0]

    # Damp overall confidence when we have little/no real history yet -
    # never report high confidence off zero (or near-zero) observations.
    experience_factor = min(model.total_observations / MIN_SAMPLES_FOR_FULL_CONFIDENCE, 1.0)
    experience_factor = max(experience_factor, 0.15)
    scale = 0.4 + 0.6 * experience_factor
    final_conf = round(top_conf * scale, 3)

    alternatives = [
        {"action": name, "confidence": round(conf * scale, 3)}
        for name, conf in ranked[1:]
    ]

    return Prediction(predicted_action=top_action, confidence=final_conf, alternatives=alternatives)

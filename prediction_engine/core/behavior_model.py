"""
core/behavior_model.py

Turns a list of HistoryEntry into a structured behavioral tendency
profile, and maintains lightweight conditional pattern tables ("what do
they do after switching", "what do they do when disadvantaged", "what
do they do after taking damage"). Pure statistics - no game knowledge.
"""

from __future__ import annotations
from typing import List
from .interfaces import HistoryEntry, OpponentModel


def compute_tendencies(history: List[HistoryEntry]) -> dict:
    """
    Compute behavioral tendencies from a full (or partial) history.
    Values are in [0, 1]. With no data, returns neutral 0.5s so
    downstream confidence stays low rather than pretending certainty.
    """
    if not history:
        return {
            "aggression": 0.5,
            "defensive_tendency": 0.5,
            "switch_tendency": 0.5,
            "repeat_action_tendency": 0.5,
        }

    n = len(history)
    attacks = sum(1 for h in history if h.action_taken.category == "attack")
    switches = sum(1 for h in history if h.action_taken.category == "switch")
    defends = sum(1 for h in history if h.action_taken.category == "defend")

    disadvantaged_turns = [h for h in history if h.was_disadvantaged]

    aggressive_when_down = (
        sum(1 for h in disadvantaged_turns if h.action_taken.category == "attack")
        / len(disadvantaged_turns)
        if disadvantaged_turns else attacks / n
    )
    aggression = 0.5 * (attacks / n) + 0.5 * aggressive_when_down

    defensive_when_down = (
        sum(1 for h in disadvantaged_turns if h.action_taken.category in ("defend", "switch"))
        / len(disadvantaged_turns)
        if disadvantaged_turns else defends / n
    )
    defensive_tendency = 0.5 * (defends / n) + 0.5 * defensive_when_down

    switch_tendency = switches / n

    repeats, comparable = 0, 0
    for h in history:
        if h.prev_action is not None:
            comparable += 1
            if h.prev_action.name == h.action_taken.name:
                repeats += 1
    repeat_action_tendency = (repeats / comparable) if comparable else 0.5

    def clamp(x: float) -> float:
        return round(min(max(x, 0.0), 1.0), 3)

    return {
        "aggression": clamp(aggression),
        "defensive_tendency": clamp(defensive_tendency),
        "switch_tendency": clamp(switch_tendency),
        "repeat_action_tendency": clamp(repeat_action_tendency),
    }


def _bump(table: dict, key: str, action_name: str) -> None:
    bucket = table.setdefault(key, {})
    bucket[action_name] = bucket.get(action_name, 0) + 1


def update_model_with_entry(model: OpponentModel, entry: HistoryEntry) -> None:
    """
    Incrementally fold one new observation into the opponent model.

    Three separate conditional tables are updated (instead of one joint
    key) so each "reaction to X" pattern stays learnable from a small
    number of observations:
      - transitions_by_prev[prev_action]      -> "attacks after switching"
      - transitions_by_disadvantage["0"/"1"]  -> "reaction to being behind"
      - transitions_by_damage["0"/"1"]        -> "reaction to taking damage"
    """
    model.history.append(entry)
    model.total_observations += 1

    action_name = entry.action_taken.name
    model.action_counts[action_name] = model.action_counts.get(action_name, 0) + 1

    prev_key = entry.prev_action.name if entry.prev_action else "none"
    _bump(model.transitions_by_prev, prev_key, action_name)
    _bump(model.transitions_by_disadvantage, str(int(entry.was_disadvantaged)), action_name)
    _bump(model.transitions_by_damage, str(int(entry.took_damage)), action_name)

    # Recomputed from full history each time - cheap at hackathon-scale
    # turn counts. Swap for a running/exponential update if history grows.
    model.tendencies = compute_tendencies(model.history)

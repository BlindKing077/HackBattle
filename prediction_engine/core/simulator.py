"""
core/simulator.py

Lightweight look-ahead: given a state and a handful of candidate
actions, use the adapter's heuristic forward-model + evaluation
function to rank them. This is NOT a full game engine (explicitly out
of scope) - one ply deep, heuristic value only.
"""

from __future__ import annotations
from typing import List, Dict
from .interfaces import GameState, Action, GameAdapter


def simulate_action(adapter: GameAdapter, state: GameState, actor_id: str, action: Action) -> float:
    """Apply one action heuristically and score the resulting state for actor_id."""
    next_state = adapter.apply_action(state, actor_id, action)
    return adapter.evaluate_state(next_state, for_actor=actor_id)


def rank_actions(
    adapter: GameAdapter,
    state: GameState,
    actor_id: str,
    candidate_actions: List[Action],
) -> List[Dict]:
    """
    Returns candidate actions sorted best-first, each with its simulated
    heuristic score, e.g.:

        [{"action": "surf", "score": 0.62}, {"action": "defend", "score": 0.1}]
    """
    results = []
    for action in candidate_actions:
        score = simulate_action(adapter, state, actor_id, action)
        results.append({"action": action.name, "score": round(score, 3)})
    return sorted(results, key=lambda r: r["score"], reverse=True)

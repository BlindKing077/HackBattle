"""
core/recommender.py

Combines opponent prediction + simulation into a single judge-readable
recommendation. This is the main entry point the backend/API calls.
"""

from __future__ import annotations
from typing import List, Optional
from .interfaces import GameState, Action, OpponentModel, GameAdapter, Recommendation, Prediction
from .predictor import predict_next_action
from .simulator import rank_actions


def recommend(
    adapter: GameAdapter,
    state: GameState,
    my_actor_id: str,
    my_actions: List[Action],
    opponent_model: OpponentModel,
    opponent_available_actions: List[Action],
    prev_opponent_action: Optional[Action],
) -> Recommendation:
    """
    1. Predict what the opponent will most likely do next.
    2. Simulate my own candidate actions and rank them.
    3. Combine into one recommendation with a human-readable reason.
    """
    was_disadv = adapter.is_disadvantaged(state)
    # Caller/adapter can enrich state.data with a real "took damage" flag;
    # defaults to False if not provided.
    took_damage = bool(state.data.get("opponent_took_damage_last_turn", False))

    opponent_prediction: Prediction = predict_next_action(
        state=state,
        model=opponent_model,
        available_actions=opponent_available_actions,
        prev_action=prev_opponent_action,
        was_disadvantaged=was_disadv,
        took_damage=took_damage,
    )

    ranked = rank_actions(adapter, state, my_actor_id, my_actions)
    if not ranked:
        return Recommendation(
            recommended_action="none",
            confidence=0.0,
            reason="No available actions to evaluate.",
            predicted_opponent_action=opponent_prediction.predicted_action,
            predicted_confidence=opponent_prediction.confidence,
        )

    best = ranked[0]
    # Confidence in *my* recommendation blends the simulated score spread
    # with how confident we are about the opponent's move - a shaky
    # opponent prediction should make us a bit less sure of our own pick.
    runner_up_score = ranked[1]["score"] if len(ranked) > 1 else best["score"] - 0.2
    score_spread = best["score"] - runner_up_score
    base_conf = min(max(0.5 + score_spread, 0.1), 0.95)
    final_conf = round(base_conf * (0.6 + 0.4 * opponent_prediction.confidence), 3)

    reason = (
        f"Opponent is predicted to use '{opponent_prediction.predicted_action}' "
        f"(confidence {opponent_prediction.confidence:.0%}). Simulating your options "
        f"against this, '{best['action']}' scores highest ({best['score']:.2f}) "
        f"among {len(ranked)} candidates."
    )

    return Recommendation(
        recommended_action=best["action"],
        confidence=final_conf,
        reason=reason,
        predicted_opponent_action=opponent_prediction.predicted_action,
        predicted_confidence=opponent_prediction.confidence,
    )

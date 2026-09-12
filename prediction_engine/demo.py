"""
demo.py

End-to-end example: opponent learning -> prediction -> simulation ->
recommendation, using the Pokemon adapter. Run from the repo root:

    python -m prediction_engine.demo
"""

from prediction_engine.core.interfaces import GameState
from prediction_engine.core.learner import OpponentLearner
from prediction_engine.core.recommender import recommend
from prediction_engine.adapters.pokemon_adapter import PokemonAdapter
from prediction_engine.tests.synthetic_data import generate_history


def build_state() -> GameState:
    return GameState(
        turn=12,
        actor_id="me",
        opponent_id="opp",
        data={
            "actors": {
                "me": {
                    "active": {"hp": 60, "max_hp": 100, "type": "water"},
                    "bench": [{"hp": 100, "max_hp": 100, "type": "grass"}],
                    "moves": [
                        {"name": "surf", "power": 90, "type": "water"},
                        {"name": "tackle", "power": 40, "type": "normal"},
                    ],
                },
                "opp": {
                    "active": {"hp": 30, "max_hp": 100, "type": "fire"},
                    "bench": [],
                    "moves": [
                        {"name": "ember", "power": 40, "type": "fire"},
                        {"name": "scratch", "power": 40, "type": "normal"},
                    ],
                },
            },
            "opponent_took_damage_last_turn": True,
        },
    )


def main():
    adapter = PokemonAdapter()
    state = build_state()

    # Warm up the opponent model with some prior observed turns (in a real
    # backend this comes from the actual match history so far).
    learner = OpponentLearner(opponent_id="opp")
    warmup = generate_history("aggressive", n_turns=15, seed=7)
    prev = None
    for turn in warmup:
        learner.observe(state, turn["action"], prev, turn["was_disadvantaged"], turn["took_damage"])
        prev = turn["action"]

    my_actions = adapter.get_available_actions(state)
    opp_perspective = GameState(turn=state.turn, actor_id="opp", opponent_id="me", data=state.data)
    opp_actions = adapter.get_available_actions(opp_perspective)

    result = recommend(
        adapter=adapter,
        state=state,
        my_actor_id="me",
        my_actions=my_actions,
        opponent_model=learner.model,
        opponent_available_actions=opp_actions,
        prev_opponent_action=prev,
    )

    print("Opponent behavior profile:", learner.model.tendencies)
    print("\nRecommendation:")
    print(f"  recommended_action        : {result.recommended_action}")
    print(f"  confidence                : {result.confidence}")
    print(f"  predicted_opponent_action : {result.predicted_opponent_action}")
    print(f"  predicted_confidence      : {result.predicted_confidence}")
    print(f"  reason                    : {result.reason}")


if __name__ == "__main__":
    main()

"""
tests/test_predictions.py

Hackathon-style smoke test - no pytest needed, run directly:

    python -m prediction_engine.tests.test_predictions

Demonstrates:
  1. The model distinguishes the 3 opponent personalities by their
     resulting tendency profiles.
  2. Prediction accuracy is tracked turn by turn as history accumulates.
  3. The "attacks after switching" pattern is specifically learned for
     the aggressive personality via the transition table.
"""

from prediction_engine.core.interfaces import GameState
from prediction_engine.core.learner import OpponentLearner
from prediction_engine.core.predictor import predict_next_action
from prediction_engine.tests.synthetic_data import generate_history, PERSONALITIES, ACTIONS


def _attack_after_switch_confidence(model) -> float:
    """Pull out the learned P(attack | just switched) for inspection."""
    counts = model.transitions_by_prev.get("switch", {})
    total = sum(counts.values())
    return round(counts.get("attack", 0) / total, 3) if total else 0.0


def run_for_personality(personality: str, n_turns: int = 60, seed: int = 1) -> dict:
    history = generate_history(personality, n_turns=n_turns, seed=seed)
    learner = OpponentLearner(opponent_id=f"demo_{personality}")

    prev_action = None
    correct_flags = []

    for i, turn in enumerate(history):
        state = GameState(turn=i, actor_id="me", opponent_id="opp")

        # Predict BEFORE seeing the real action - using only what we know so far.
        prediction = predict_next_action(
            state=state,
            model=learner.model,
            available_actions=ACTIONS,
            prev_action=prev_action,
            was_disadvantaged=turn["was_disadvantaged"],
            took_damage=turn["took_damage"],
        )
        learner.record_prediction(prediction)

        report = learner.observe(
            state=state,
            action_taken=turn["action"],
            prev_action=prev_action,
            was_disadvantaged=turn["was_disadvantaged"],
            took_damage=turn["took_damage"],
        )
        if "correct" in report:
            correct_flags.append(report["correct"])
        prev_action = turn["action"]

    last_10 = correct_flags[-10:]
    return {
        "personality": personality,
        "final_tendencies": learner.model.tendencies,
        "overall_accuracy": round(learner.model.accuracy, 3),
        "accuracy_last_10_turns": round(sum(last_10) / len(last_10), 3) if last_10 else 0.0,
        "attack_after_switch_confidence": _attack_after_switch_confidence(learner.model),
    }


def main():
    print("=" * 70)
    print("AI Battle Intelligence - opponent modeling smoke test")
    print("=" * 70)

    results = [run_for_personality(p) for p in PERSONALITIES]
    for r in results:
        print(f"\n[{r['personality'].upper()}]")
        print(f"  tendencies                 : {r['final_tendencies']}")
        print(f"  overall prediction accuracy: {r['overall_accuracy']:.0%}")
        print(f"  accuracy on last 10 turns  : {r['accuracy_last_10_turns']:.0%}")
        print(f"  P(attack | just switched)  : {r['attack_after_switch_confidence']:.0%}")

    print("\nNote: 'unpredictable' is generated close to uniform-random across 3\n"
          "actions on purpose, so ~33% accuracy is close to its theoretical\n"
          "ceiling - no model should beat random on a genuinely random opponent.\n"
          "That the system doesn't overclaim there either is the point.")

    print("\n" + "-" * 70)
    print("Sanity checks:")
    by_p = {r["personality"]: r for r in results}

    assert by_p["aggressive"]["final_tendencies"]["aggression"] > by_p["defensive"]["final_tendencies"]["aggression"], \
        "aggressive should score higher aggression than defensive"
    assert by_p["defensive"]["final_tendencies"]["defensive_tendency"] > by_p["aggressive"]["final_tendencies"]["defensive_tendency"], \
        "defensive should score higher defensive_tendency than aggressive"
    assert by_p["aggressive"]["attack_after_switch_confidence"] > 0.6, \
        "should have learned 'attacks after switching' for the aggressive opponent"

    print("All checks passed: model distinguishes personalities and learns the switch -> attack pattern.")


if __name__ == "__main__":
    main()

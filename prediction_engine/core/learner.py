"""
core/learner.py

Ties observation -> model update -> accuracy tracking together. This is
the object the backend calls once per real observed turn.
"""

from __future__ import annotations
from typing import Optional
from .interfaces import OpponentModel, HistoryEntry, Action, GameState, Prediction
from .behavior_model import update_model_with_entry


class OpponentLearner:
    """
    One instance per opponent. Call `record_prediction` right before the
    opponent acts (optional, but needed for accuracy tracking), then
    call `observe` once you know what they actually did.
    """

    def __init__(self, opponent_id: str):
        self.model = OpponentModel(opponent_id=opponent_id)
        self._pending_prediction: Optional[Prediction] = None

    def record_prediction(self, prediction: Prediction) -> None:
        """Stash the prediction so it can be scored once the real action is known."""
        self._pending_prediction = prediction

    def observe(
        self,
        state: GameState,
        action_taken: Action,
        prev_action: Optional[Action],
        was_disadvantaged: bool,
        took_damage: bool,
    ) -> dict:
        """
        Record what actually happened, update the model, and score the
        pending prediction (if any). Returns a small report dict, handy
        for demo logging / a frontend event feed.
        """
        entry = HistoryEntry(
            turn=state.turn,
            state=state,
            action_taken=action_taken,
            was_disadvantaged=was_disadvantaged,
            took_damage=took_damage,
            prev_action=prev_action,
        )

        report: dict = {"turn": state.turn, "actual": action_taken.name}

        if self._pending_prediction is not None:
            predicted = self._pending_prediction.predicted_action
            correct = predicted == action_taken.name
            self.model.predictions_made += 1
            if correct:
                self.model.predictions_correct += 1
            report["predicted"] = predicted
            report["predicted_confidence"] = self._pending_prediction.confidence
            report["correct"] = correct
            self._pending_prediction = None

        update_model_with_entry(self.model, entry)
        report["accuracy_so_far"] = round(self.model.accuracy, 3)
        report["tendencies"] = dict(self.model.tendencies)
        return report

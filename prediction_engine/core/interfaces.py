"""
core/interfaces.py

Game-independent data structures and the adapter contract for the
AI Battle Intelligence prediction system.

Any new game (Pokemon, FPS, fighting game, strategy game, ...) plugs in
by implementing GameAdapter. Nothing else in core/ should ever import a
game-specific module.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod


@dataclass
class Action:
    """A single possible move/action in the game."""
    name: str                     # e.g. "flamethrower", "switch", "defend"
    category: str                 # generic bucket: "attack" | "switch" | "defend" | "other"
    power: float = 0.0            # rough expected impact, adapter-normalized
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)


@dataclass
class GameState:
    """
    Generic snapshot of the game at a point in time. `data` is a
    free-form payload the adapter fills in (HP, type matchups,
    positions, ammo, whatever the game needs). Core modules never read
    `data` directly except through the adapter.
    """
    turn: int
    actor_id: str          # whose turn / perspective this state is framed from
    opponent_id: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HistoryEntry:
    """One observed turn, used for learning."""
    turn: int
    state: GameState
    action_taken: Action
    was_disadvantaged: bool = False   # opponent was at a disadvantage before acting
    took_damage: bool = False         # opponent took damage on the previous turn
    prev_action: Optional[Action] = None


@dataclass
class OpponentModel:
    """Structured behavioral profile + learning state for one opponent."""
    opponent_id: str
    tendencies: Dict[str, float] = field(default_factory=lambda: {
        "aggression": 0.5,
        "defensive_tendency": 0.5,
        "switch_tendency": 0.5,
        "repeat_action_tendency": 0.5,
    })
    # Conditional pattern tables, each: bucket_key -> {action_name: count}.
    # Kept separate (rather than one big joint key) so each stays sample-
    # efficient: "reaction to X" patterns are learnable from a handful of
    # observations instead of needing every combination to repeat.
    transitions_by_prev: Dict[str, Dict[str, int]] = field(default_factory=dict)        # prev action -> next action
    transitions_by_disadvantage: Dict[str, Dict[str, int]] = field(default_factory=dict)  # "0"/"1" -> action
    transitions_by_damage: Dict[str, Dict[str, int]] = field(default_factory=dict)        # "0"/"1" -> action
    action_counts: Dict[str, int] = field(default_factory=dict)
    history: List[HistoryEntry] = field(default_factory=list)
    total_observations: int = 0

    # accuracy tracking
    predictions_made: int = 0
    predictions_correct: int = 0

    @property
    def accuracy(self) -> float:
        if self.predictions_made == 0:
            return 0.0
        return self.predictions_correct / self.predictions_made


@dataclass
class Prediction:
    predicted_action: str
    confidence: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Recommendation:
    recommended_action: str
    confidence: float
    reason: str
    predicted_opponent_action: str
    predicted_confidence: float


class GameAdapter(ABC):
    """
    Contract every game plugs in through. The prediction engine only
    ever talks to a GameAdapter, never to Pokemon/FPS/etc. directly.
    """

    @abstractmethod
    def get_available_actions(self, state: GameState) -> List[Action]:
        """Return the actions the actor could take from this state."""

    @abstractmethod
    def is_disadvantaged(self, state: GameState) -> bool:
        """Is the opponent at a disadvantage right now (low HP, bad matchup, etc)?"""

    @abstractmethod
    def evaluate_state(self, state: GameState, for_actor: str) -> float:
        """
        Lightweight heuristic value of a state for `for_actor`, roughly
        -1..1. Used by the simulator to rank hypothetical outcomes. It
        does not need to be exact, just directionally useful for a demo.
        """

    @abstractmethod
    def apply_action(self, state: GameState, actor_id: str, action: Action) -> GameState:
        """
        Return a *new* GameState approximating what happens if `actor_id`
        takes `action`. This is a heuristic forward-model, not a full
        game engine.
        """

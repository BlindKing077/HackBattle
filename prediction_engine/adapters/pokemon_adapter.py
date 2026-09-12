"""
adapters/pokemon_adapter.py

Pokemon-specific glue. This is the ONLY file that should know anything
about types, HP, or moves. Everything in core/ stays game-agnostic.

Deliberately simplified type chart + damage model - good enough for a
hackathon demo, not meant to be battle-accurate. Expected `state.data`
shape:

    {
        "actors": {
            "<actor_id>": {
                "active": {"hp": float, "max_hp": float, "type": str},
                "bench": [{"hp": ..., "max_hp": ..., "type": ...}, ...],
                "moves": [{"name": str, "power": float, "type": str}, ...],
            },
            "<opponent_id>": {...same shape...},
        },
        "opponent_took_damage_last_turn": bool,   # optional, used by recommender
    }
"""

from __future__ import annotations
import copy
from typing import List, Dict
from prediction_engine.core.interfaces import GameAdapter, GameState, Action

# Small type chart: attacker_type -> {defender_type: multiplier}. Only
# covers a handful of types - enough for a hackathon demo.
TYPE_CHART: Dict[str, Dict[str, float]] = {
    "fire":     {"grass": 2.0, "water": 0.5, "fire": 0.5},
    "water":    {"fire": 2.0, "grass": 0.5, "water": 0.5},
    "grass":    {"water": 2.0, "fire": 0.5, "grass": 0.5},
    "electric": {"water": 2.0, "grass": 0.5, "electric": 0.5},
    "normal":   {},
}


class PokemonAdapter(GameAdapter):
    def get_available_actions(self, state: GameState) -> List[Action]:
        actor = state.data["actors"][state.actor_id]
        actions = []
        for move in actor["moves"]:
            actions.append(Action(
                name=move["name"],
                category="attack",
                power=move.get("power", 40) / 100.0,
                metadata={"type": move.get("type", "normal")},
            ))
        if actor.get("bench"):
            actions.append(Action(name="switch", category="switch"))
        actions.append(Action(name="defend", category="defend"))
        return actions

    def is_disadvantaged(self, state: GameState) -> bool:
        opp = state.data["actors"][state.opponent_id]["active"]
        me = state.data["actors"][state.actor_id]["active"]
        hp_ratio = opp["hp"] / opp["max_hp"]
        type_mult = self._effectiveness(me["type"], opp["type"])
        return hp_ratio < 0.35 or type_mult >= 2.0

    def evaluate_state(self, state: GameState, for_actor: str) -> float:
        other = state.opponent_id if for_actor == state.actor_id else state.actor_id
        me = state.data["actors"][for_actor]["active"]
        opp = state.data["actors"][other]["active"]
        me_ratio = me["hp"] / me["max_hp"]
        opp_ratio = opp["hp"] / opp["max_hp"]

        # A knockout dominates any HP-ratio bookkeeping: finishing the
        # opponent off this turn beats "having more HP in reserve", and
        # getting knocked out ourselves is the worst outcome regardless
        # of what's left on the bench.
        if opp_ratio <= 0.0 and me_ratio > 0.0:
            return 1.0
        if me_ratio <= 0.0:
            return -1.0
        return round(me_ratio - opp_ratio, 3)  # roughly -1..1

    def apply_action(self, state: GameState, actor_id: str, action: Action) -> GameState:
        new_state = copy.deepcopy(state)
        actors = new_state.data["actors"]
        other_id = new_state.opponent_id if actor_id == new_state.actor_id else new_state.actor_id

        if action.category == "attack":
            attacker = actors[actor_id]["active"]
            defender = actors[other_id]["active"]
            mult = self._effectiveness(action.metadata.get("type", "normal"), defender["type"])
            dmg = action.power * 40 * mult  # heuristic, not real Pokemon damage math
            defender["hp"] = max(0.0, defender["hp"] - dmg)
        elif action.category == "switch":
            bench = actors[actor_id].get("bench")
            if bench:
                actors[actor_id]["active"], bench[0] = bench[0], actors[actor_id]["active"]
        # "defend" -> no immediate state change in this simplified model

        return new_state

    @staticmethod
    def _effectiveness(attack_type: str, defend_type: str) -> float:
        return TYPE_CHART.get(attack_type, {}).get(defend_type, 1.0)

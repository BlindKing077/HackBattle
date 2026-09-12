# prediction_engine

Prediction + Learning + Simulation module for **AI Battle Intelligence**.
Owns opponent modeling, next-action prediction, learning from outcomes,
lightweight look-ahead simulation, and the final recommendation. It does
**not** contain a frontend, a backend API, or Pokemon-only logic — all
game-specific code is isolated in `adapters/`.

## 1. Files created

```
prediction_engine/
├── __init__.py                  package entry point / integration docstring
├── core/
│   ├── interfaces.py             GameState, Action, OpponentModel, Prediction,
│   │                              Recommendation, GameAdapter (the generic contract)
│   ├── behavior_model.py         turns raw history into tendency scores +
│   │                              conditional pattern tables
│   ├── predictor.py               next-action prediction engine
│   ├── learner.py                 observe -> update model -> track accuracy
│   ├── simulator.py               one-ply heuristic look-ahead / ranking
│   └── recommender.py             combines prediction + simulation -> Recommendation
├── adapters/
│   └── pokemon_adapter.py        the ONLY file that knows about Pokemon
│                                  types/HP/moves; implements GameAdapter
├── tests/
│   ├── synthetic_data.py         3 synthetic opponent personalities
│   └── test_predictions.py       smoke test proving learning + distinction
├── demo.py                        end-to-end example (prediction -> sim -> recommendation)
└── README.md                      this file
```

## 2. What each file does

- **`core/interfaces.py`** — the game-independent vocabulary. `GameState`,
  `Action`, `OpponentModel`, `Prediction`, `Recommendation` are plain
  dataclasses. `GameAdapter` is an abstract base class every game
  implements (`get_available_actions`, `is_disadvantaged`,
  `evaluate_state`, `apply_action`). Nothing in `core/` imports a game
  module — this is what makes the engine game-independent.
- **`core/behavior_model.py`** — computes `aggression`,
  `defensive_tendency`, `switch_tendency`, `repeat_action_tendency` from
  raw history, and maintains three small conditional pattern tables
  (reaction to previous action, to being disadvantaged, to taking
  damage). Kept as three separate tables instead of one big joint key
  so each pattern stays learnable from a handful of observations.
- **`core/predictor.py`** — predicts the opponent's next action by
  blending those conditional tables with the general tendencies,
  weighting each signal by how much evidence backs it. Confidence is
  explicitly capped when total observations are low.
- **`core/learner.py`** — `OpponentLearner`: call once per real observed
  turn. Records the prediction that was made, checks it against what
  actually happened, updates the model, tracks running accuracy.
- **`core/simulator.py`** — one-ply heuristic look-ahead: applies each
  candidate action via the adapter's `apply_action`, scores the result
  via `evaluate_state`, ranks the candidates. Not a full game engine on
  purpose.
- **`core/recommender.py`** — `recommend(...)`: the main entry point.
  Predicts the opponent's next move, ranks your own candidate actions
  against that, and returns a `Recommendation` with a plain-English
  `reason` a judge can read.
- **`adapters/pokemon_adapter.py`** — implements `GameAdapter` for
  Pokemon: a small type chart, simplified damage formula, and knockout
  handling. Swap this file out (FPS/fighting-game/strategy-game
  adapters) without touching anything in `core/`.
- **`tests/synthetic_data.py`** — generates believable turn-by-turn
  histories for 3 opponent personalities (aggressive, defensive,
  unpredictable). The aggressive personality deliberately attacks right
  after switching, so the pattern table has something concrete to learn.
- **`tests/test_predictions.py`** — runs the learner turn-by-turn over
  each personality and asserts the model tells them apart and picks up
  the switch→attack pattern.
- **`demo.py`** — wires everything together with the Pokemon adapter for
  one realistic mid-battle scenario.

## 3. How to run

No third-party dependencies — standard library only. From the **repo
root** (the folder containing `prediction_engine/`):

```bash
python3 -m prediction_engine.tests.test_predictions
python3 -m prediction_engine.demo
```

## 4. Example input / output

**Prediction** (`core/predictor.predict_next_action`):

```
Input:  opponent just switched, is not disadvantaged, took no damage,
        available actions = [attack, switch, defend]
Output: Prediction(predicted_action="attack", confidence=0.81,
                    alternatives=[{"action": "switch", "confidence": 0.14},
                                  {"action": "defend", "confidence": 0.05}])
```

**Recommendation** (`core/recommender.recommend`), from `demo.py`:

```json
{
  "recommended_action": "surf",
  "confidence": 0.652,
  "reason": "Opponent is predicted to use 'defend' (confidence 54%). Simulating your options against this, 'surf' scores highest (1.00) among 4 candidates.",
  "predicted_opponent_action": "defend",
  "predicted_confidence": 0.539
}
```

## 5. How the backend should call this

The backend never touches `core/` internals directly — it goes through
an adapter + the two top-level entry points, `OpponentLearner` and
`recommend`:

```python
from prediction_engine.adapters.pokemon_adapter import PokemonAdapter
from prediction_engine.core.learner import OpponentLearner
from prediction_engine.core.recommender import recommend

adapter = PokemonAdapter()
learner = OpponentLearner(opponent_id="opp_123")   # one per active opponent

# Each time the backend sees the opponent actually act:
learner.observe(
    state=current_state,          # GameState built from the live match
    action_taken=action_they_took,  # Action they actually took
    prev_action=their_last_action,
    was_disadvantaged=adapter.is_disadvantaged(current_state),
    took_damage=opponent_took_damage_this_turn,   # backend/game state tracks this
)

# Whenever the frontend needs a recommendation for the current turn:
result = recommend(
    adapter=adapter,
    state=current_state,
    my_actor_id="me",
    my_actions=adapter.get_available_actions(current_state),
    opponent_model=learner.model,
    opponent_available_actions=adapter.get_available_actions(opponent_perspective_state),
    prev_opponent_action=their_last_action,
)
# result.recommended_action, result.confidence, result.reason,
# result.predicted_opponent_action, result.predicted_confidence
```

`OpponentModel` (`learner.model`) is a plain dataclass — call
`dataclasses.asdict(learner.model)` if the API needs to serialize the
full behavioral profile for the frontend.

## 6. Dependencies

None beyond the Python 3 standard library. No `pip install` needed.

## 7. Plugging in a new game later

Implement `GameAdapter` (see `adapters/pokemon_adapter.py` as the
template) for FPS / fighting-game / strategy-game and pass it into the
same `OpponentLearner` / `recommend` calls above — nothing in `core/`
changes.

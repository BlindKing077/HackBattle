"""
AI Battle Intelligence - prediction / learning / simulation engine.

Game-independent by design: everything in core/ only knows about the
generic GameState / Action / OpponentModel / Prediction / Recommendation
types defined in core/interfaces.py. Game-specific logic (Pokemon, FPS,
fighting games, ...) lives entirely in adapters/.

Backend integration entry point:

    from prediction_engine.core.learner import OpponentLearner
    from prediction_engine.core.recommender import recommend
    from prediction_engine.adapters.pokemon_adapter import PokemonAdapter

See README.md in this folder for the full integration contract.
"""

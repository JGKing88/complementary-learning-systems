"""Rollout layer: driving a policy through an environment and recording it.

`collector.py` is the Hopfield stack's loop, `rnn.py` the RNN baseline's; they
share only the environment, deliberately. `signal.py`, `oracles.py` and
`distractors.py` are the pieces both the collector and the evaluators need, and
`types.py` holds the record a rollout produces.

May import from world and policy. Must not import from updates, evaluation,
training or analysis.
"""

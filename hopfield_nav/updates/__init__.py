"""Update layer: the loss functions and optimizer steps.

PPO, behavior cloning, and the RNN baseline's BC. Each consumes a rollout
record and mutates an agent; none of them collects data or evaluates.

May import from world, policy and rollout. Must not import from evaluation,
training or analysis.
"""

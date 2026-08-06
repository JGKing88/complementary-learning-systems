"""Policy layer: the networks that map an observation to an action.

Owns the policy-input *layout* as well as the networks (`channels.py`), because
`agent.compute_input_dim` derives the width the network is built with from the
same channel specs the observation is assembled from -- if those two ever
disagree the tensor still has the right shape and only the meaning of the
channels changes, which nothing raises on.

May import from world. Must not import from rollout, updates, evaluation,
training or analysis.
"""

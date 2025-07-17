import torch as th
import numpy as np
from gymnasium import spaces

from uav_intent_rl.policies.amf_policy import AMFPolicy


def test_amf_policy_shapes():
    """Sanity-check that AMFPolicy compiles and returns correct shapes."""

    obs_dim = 12
    action_dim = 4

    # Dummy continuous spaces (match PyBullet drones env)
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    policy = AMFPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 3e-4,
    )

    batch = 5
    obs_tensor = th.zeros((batch, obs_dim))

    # Standard forward (used by SB3 internally)
    actions, values, log_prob = policy(obs_tensor)
    assert actions.shape == (batch, action_dim)
    assert values.shape == (batch, 1)
    assert log_prob.shape == (batch,)

    # Extended forward that exposes h_opp
    actions2, values2, h_opp = policy.policy_forward(obs_tensor)
    assert actions2.shape == (batch, action_dim)
    assert values2.shape == (batch, 1)
    assert h_opp.shape == (batch, 32) 
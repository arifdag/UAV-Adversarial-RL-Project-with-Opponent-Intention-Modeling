from __future__ import annotations

"""Custom Actor-Critic policy with an auxiliary head that predicts the red drone’s next
(discretised) action bucket.

The policy subclasses ``stable_baselines3.common.policies.ActorCriticPolicy`` and
adds a fully-connected layer ``opp_head`` whose output dimension equals the number
of discrete action buckets (default **5** – see :pyfunc:`bucketize_red_action`).
During optimisation the IntentPPO algorithm adds a cross-entropy term between the
predicted logits and the ground-truth bucket id provided by the environment.
"""

from typing import Any, Dict

import torch as th
from torch import nn
# Stable-Baselines3 ≥1.8 ships ``register_policy`` but earlier versions do not.
# Import it defensively so that the code also works on those versions.
from stable_baselines3.common.policies import ActorCriticPolicy

try:
    from stable_baselines3.common.policies import register_policy  # type: ignore
except ImportError:  # pragma: no cover – SB3 version without helper
    register_policy = None  # type: ignore

__all__ = ["IntentPPOPolicy"]


class IntentPPOPolicy(ActorCriticPolicy):
    """Actor-Critic network with an auxiliary opponent-modelling head."""

    def __init__(
        self,
        *args: Any,
        opp_num_buckets: int = 5,
        **kwargs: Any,
    ) -> None:
        self.opp_num_buckets = opp_num_buckets
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Network construction – override to inject extra head
    # ------------------------------------------------------------------

    def _build_mlp_extractor(self) -> None:  # noqa: D401 – SB3 internal name
        """Extend base extractor by adding :pyattr:`opp_head`."""
        super()._build_mlp_extractor()

        latent_dim = self.mlp_extractor.latent_dim_pi
        # Increase capacity: two-layer MLP (latent → hidden → logits)
        hidden_dim = max(64, latent_dim)
        self.opp_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.opp_num_buckets),
        )

    # ------------------------------------------------------------------
    # Convenience helper – return opponent logits for a batch of obs
    # ------------------------------------------------------------------

    def opponent_logits(self, obs: th.Tensor) -> th.Tensor:  # noqa: D401
        """Compute raw (unnormalised) logits for opponent-action buckets."""
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        return self.opp_head(latent_pi)


# ---------------------------------------------------------------------
# Register so SB3 can load via string "IntentPPOPolicy"
# ---------------------------------------------------------------------

# Only register with SB3 if the helper exists in the installed version.
if register_policy is not None:  # pragma: no cover
    register_policy("IntentPPOPolicy", IntentPPOPolicy) 
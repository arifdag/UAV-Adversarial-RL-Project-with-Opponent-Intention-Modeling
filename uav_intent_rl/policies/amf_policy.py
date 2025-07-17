from __future__ import annotations

"""Actor-Critic policy with AMF-style opponent-aware latent fusion.

This network augments the standard *ActorCriticPolicy* from
Stable-Baselines3 with an auxiliary *opponent head* that predicts a
**discrete** action bucket for the *red* drone while also emitting a
32-D latent feature vector (``h_opp``).  The feature vector is
*concatenated* with the agent latent before the actor/critic heads –
as proposed in the **AMF** architecture (He et al., 2022).

Training uses the standard PPO loss **plus** a λ-weighted
cross-entropy term on the logits returned by
:pyfunc:`opponent_logits` (implemented by
*uav_intent_rl.algo.intent_ppo.IntentPPO*).
"""

from typing import Any, Tuple

import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

try:
    # Only available in SB3 ≥1.8
    from stable_baselines3.common.policies import register_policy  # type: ignore
except ImportError:  # pragma: no cover – older SB3
    register_policy = None  # type: ignore

__all__: list[str] = ["AMFPolicy"]


class AMFPolicy(ActorCriticPolicy):
    """PPO policy with auxiliary opponent-feature head.

    Extra keyword arguments
    -----------------------
    opp_num_buckets : int, default **5**
        Number of discrete action buckets for the opponent (*red*) drone.
    opp_feature_dim : int, default **32**
        Dimensionality of the latent feature ``h_opp`` that gets fused
        into the policy/value heads.
    """

    def __init__(
        self,
        *args: Any,
        opp_num_buckets: int = 5,
        opp_feature_dim: int = 32,
        **kwargs: Any,
    ) -> None:
        self.opp_num_buckets = int(opp_num_buckets)
        self.opp_feature_dim = int(opp_feature_dim)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Network construction – inject opponent head & enlarge actor/critic
    # ------------------------------------------------------------------

    def _build_mlp_extractor(self) -> None:  # noqa: D401 – SB3 internal name
        """Extend parent extractor and add custom AMF heads."""
        super()._build_mlp_extractor()

        latent_dim_pi: int = self.mlp_extractor.latent_dim_pi  # type: ignore[attr-defined]
        latent_dim_vf: int = self.mlp_extractor.latent_dim_vf  # type: ignore[attr-defined]

        # Opponent feature extractor
        hidden_dim = max(64, latent_dim_pi)
        self.opp_feat = nn.Sequential(
            nn.Linear(latent_dim_pi, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.opp_feature_dim),
            nn.ReLU(),
        )
        self.opp_cls = nn.Linear(self.opp_feature_dim, self.opp_num_buckets)

        # Create new heads for fused features. This avoids fighting the parent's
        # _build() method, which overwrites self.action_net.
        fused_dim_pi = latent_dim_pi + self.opp_feature_dim
        fused_dim_vf = latent_dim_vf + self.opp_feature_dim

        if isinstance(self.action_space, spaces.Box):
            act_out = int(self.action_space.shape[0])  # type: ignore[arg-type]
        else:  # Discrete
            act_out = int(self.action_space.n)  # type: ignore[attr-defined]

        self.amf_action_net = nn.Linear(fused_dim_pi, act_out)
        self.amf_value_net = nn.Linear(fused_dim_vf, 1)

    # ------------------------------------------------------------------
    # Helper – logits for CE loss (called by IntentPPO)
    # ------------------------------------------------------------------

    def opponent_logits(self, obs: th.Tensor) -> th.Tensor:  # noqa: D401
        """Return *raw* logits for red action-bucket classification."""
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        h_opp = self.opp_feat(latent_pi)
        logits = self.opp_cls(h_opp)
        return logits

    # ------------------------------------------------------------------
    # Forward pass – returns (actions, values, log_prob)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:  # type: ignore[override]
        """Compute action, value & log-prob **with** opponent fusion."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        h_opp = self.opp_feat(latent_pi)
        fused_pi = th.cat([latent_pi, h_opp], dim=-1)
        fused_vf = th.cat([latent_vf, h_opp], dim=-1)

        # Use our custom AMF heads, replicating parent's distribution logic
        mean_actions = self.amf_action_net(fused_pi)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.amf_value_net(fused_vf)

        return actions, values, log_prob

    # ------------------------------------------------------------------
    # Convenience API – expose h_opp for downstream analysis/logging
    # ------------------------------------------------------------------

    def policy_forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Return *(action, value, h_opp)* as required by the spec."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        h_opp = self.opp_feat(latent_pi)
        fused_pi = th.cat([latent_pi, h_opp], dim=-1)
        fused_vf = th.cat([latent_vf, h_opp], dim=-1)

        mean_actions = self.amf_action_net(fused_pi)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self.amf_value_net(fused_vf)
        return actions, values, h_opp

    # ------------------------------------------------------------------
    # Override evaluate_actions to use fused latents during optimisation
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:  # type: ignore[override]
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        h_opp = self.opp_feat(latent_pi)
        fused_pi = th.cat([latent_pi, h_opp], dim=-1)
        fused_vf = th.cat([latent_vf, h_opp], dim=-1)

        mean_actions = self.amf_action_net(fused_pi)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.amf_value_net(fused_vf)
        return values, log_prob, entropy


# ---------------------------------------------------------------------------
# Register with SB3 so it can be referenced by the string "AMFPolicy"
# ---------------------------------------------------------------------------

if register_policy is not None:  # pragma: no cover
    register_policy("AMFPolicy", AMFPolicy) 
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
    """PPO policy with auxiliary opponent-feature head and optional LSTM/GRU support."""

    def __init__(
        self,
        *args: Any,
        opp_num_buckets: int = 5,
        opp_feature_dim: int = 32,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 1,
        lstm_type: str = "lstm",  # or "gru"
        use_lstm: bool = False,
        **kwargs: Any,
    ) -> None:
        self.opp_num_buckets = int(opp_num_buckets)
        self.opp_feature_dim = int(opp_feature_dim)
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_type = lstm_type.lower()
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Network construction – inject opponent head & enlarge actor/critic
    # ------------------------------------------------------------------

    def _build_mlp_extractor(self) -> None:  # noqa: D401 – SB3 internal name
        """Extend parent extractor and add custom AMF heads."""
        super()._build_mlp_extractor()

        latent_dim_pi: int = self.mlp_extractor.latent_dim_pi  # type: ignore[attr-defined]
        latent_dim_vf: int = self.mlp_extractor.latent_dim_vf  # type: ignore[attr-defined]

        # Optional LSTM/GRU after MLP extractor
        if self.use_lstm:
            rnn_cls = nn.LSTM if self.lstm_type == "lstm" else nn.GRU
            self.rnn = rnn_cls(
                input_size=latent_dim_pi,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_num_layers,
                batch_first=True,
            )
            rnn_out_dim = self.lstm_hidden_size
        else:
            self.rnn = None
            rnn_out_dim = latent_dim_pi

        # Opponent feature extractor
        hidden_dim = max(64, rnn_out_dim)
        self.opp_feat = nn.Sequential(
            nn.Linear(rnn_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.opp_feature_dim),
            nn.ReLU(),
        )
        self.opp_cls = nn.Linear(self.opp_feature_dim, self.opp_num_buckets)

        # Create new heads for fused features. This avoids fighting the parent's
        # _build() method, which overwrites self.action_net.
        fused_dim_pi = rnn_out_dim + self.opp_feature_dim
        fused_dim_vf = rnn_out_dim + self.opp_feature_dim

        if isinstance(self.action_space, spaces.Box):
            act_out = int(self.action_space.shape[0])  # type: ignore[arg-type]
        else:  # Discrete
            act_out = int(self.action_space.n)  # type: ignore[attr-defined]

        self.amf_action_net = nn.Linear(fused_dim_pi, act_out)
        self.amf_value_net = nn.Linear(fused_dim_vf, 1)

    # ------------------------------------------------------------------
    # Helper – logits for CE loss (called by IntentPPO)
    # ------------------------------------------------------------------

    def _reset_hidden_state(self, state, episode_starts):
        # Handles both LSTM (tuple) and GRU (tensor)
        if state is None or episode_starts is None:
            return state
        import torch as th
        mask = th.as_tensor(episode_starts).bool()
        if isinstance(state, tuple):  # LSTM: (h, c)
            h, c = state
            if mask.dim() == 1:
                h[:, mask, :] = 0.0
                c[:, mask, :] = 0.0
            else:
                # If mask is [batch, seq], reset at first step
                h[:, mask[:, 0], :] = 0.0
                c[:, mask[:, 0], :] = 0.0
            return (h, c)
        else:  # GRU: tensor
            if mask.dim() == 1:
                state[:, mask, :] = 0.0
            else:
                state[:, mask[:, 0], :] = 0.0
            return state

    def _process_sequence(self, latent, state=None, episode_starts=None):
        # latent: [batch, features] or [batch, seq, features]
        if not self.use_lstm:
            return latent, state
        if latent.dim() == 2:
            latent = latent.unsqueeze(1)  # [batch, 1, features]
        if state is not None and episode_starts is not None:
            state = self._reset_hidden_state(state, episode_starts)
        rnn_out, new_state = self.rnn(latent, state)
        if rnn_out.shape[1] == 1:
            rnn_out = rnn_out.squeeze(1)
        return rnn_out, new_state

    def opponent_logits(self, obs: th.Tensor, state=None, episode_starts=None):
        """Return *raw* logits for red action-bucket classification."""
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        latent_pi, _ = self._process_sequence(latent_pi, state, episode_starts)
        h_opp = self.opp_feat(latent_pi)
        logits = self.opp_cls(h_opp)
        return logits

    # ------------------------------------------------------------------
    # Forward pass – returns (actions, values, log_prob)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: th.Tensor,
        state=None,
        episode_starts=None,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Any]:  # type: ignore[override]
        """Compute action, value & log-prob **with** opponent fusion."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, state = self._process_sequence(latent_pi, state, episode_starts)
        latent_vf, _ = self._process_sequence(latent_vf, state, episode_starts) if self.use_lstm else (latent_vf, state)
        h_opp = self.opp_feat(latent_pi)
        fused_pi = th.cat([latent_pi, h_opp], dim=-1)
        fused_vf = th.cat([latent_vf, h_opp], dim=-1)

        # Use our custom AMF heads, replicating parent's distribution logic
        mean_actions = self.amf_action_net(fused_pi)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.amf_value_net(fused_vf)

        return actions, values, log_prob, state

    # ------------------------------------------------------------------
    # Convenience API – expose h_opp for downstream analysis/logging
    # ------------------------------------------------------------------

    def policy_forward(
        self,
        obs: th.Tensor,
        state=None,
        episode_starts=None,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Any]:
        """Return *(action, value, h_opp)* as required by the spec."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, state = self._process_sequence(latent_pi, state, episode_starts)
        latent_vf, _ = self._process_sequence(latent_vf, state, episode_starts) if self.use_lstm else (latent_vf, state)
        h_opp = self.opp_feat(latent_pi)
        fused_pi = th.cat([latent_pi, h_opp], dim=-1)
        fused_vf = th.cat([latent_vf, h_opp], dim=-1)

        mean_actions = self.amf_action_net(fused_pi)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self.amf_value_net(fused_vf)
        return actions, values, h_opp, state

    # ------------------------------------------------------------------
    # Override evaluate_actions to use fused latents during optimisation
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        state=None,
        episode_starts=None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, Any]:  # type: ignore[override]
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        latent_pi, state = self._process_sequence(latent_pi, state, episode_starts)
        latent_vf, _ = self._process_sequence(latent_vf, state, episode_starts) if self.use_lstm else (latent_vf, state)
        h_opp = self.opp_feat(latent_pi)
        fused_pi = th.cat([latent_pi, h_opp], dim=-1)
        fused_vf = th.cat([latent_vf, h_opp], dim=-1)

        mean_actions = self.amf_action_net(fused_pi)
        distribution = self.action_dist.proba_distribution(mean_actions, self.log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.amf_value_net(fused_vf)
        return values, log_prob, entropy, state


# ---------------------------------------------------------------------------
# Register with SB3 so it can be referenced by the string "AMFPolicy"
# ---------------------------------------------------------------------------

if register_policy is not None:  # pragma: no cover
    register_policy("AMFPolicy", AMFPolicy) 
from __future__ import annotations

"""IntentPPO – PPO with an auxiliary opponent-prediction loss.

The algorithm is a light wrapper around :class:`stable_baselines3.ppo.PPO`
which (1) uses :class:`~uav_intent_rl.policies.intent_ppo_policy.IntentPPOPolicy`
by default, (2) collects the *red* drone’s discrete action bucket at every
environment step and stores it in a custom rollout buffer, and (3) adds a
cross-entropy term to the standard PPO objective:

    L_total = L_PPO + λ · CE(opp_logits, red_bucket)

where *λ* is a user-supplied hyper-parameter (``aux_loss_coef``).
"""

from typing import Any, Optional, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer, RolloutBufferSamples
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.ppo.ppo import PPO

from uav_intent_rl.policies.intent_ppo_policy import IntentPPOPolicy

__all__ = ["IntentPPO"]


# ---------------------------------------------------------------------------
# Helper – discretise red's continuous velocity action into simple buckets
# ---------------------------------------------------------------------------

def bucketize_red_action(action: np.ndarray) -> int:
    """Map a 4-D continuous velocity **action** to one of 5 buckets.

    Bucket definition (based on XY direction only):
    0 – near-zero XY movement  (‖v‖ < 0.3)
    1 – NE
    2 – NW
    3 – SW
    4 – SE
    """
    x, y = float(action[0]), float(action[1])
    if abs(x) <= 0.3 and abs(y) <= 0.3:
        return 0
    if x >= 0 and y >= 0:
        return 1  # NE
    if x < 0 and y >= 0:
        return 2  # NW
    if x < 0 and y < 0:
        return 3  # SW
    return 4  # SE


# ---------------------------------------------------------------------------
# Rollout buffer with extra *opp_actions* field
# ---------------------------------------------------------------------------

class RolloutBufferWithOpp(RolloutBuffer):
    """Extends SB3 RolloutBuffer to store opponent-action bucket ids."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Buffer shape: (n_steps, n_envs)
        self.opp_actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)

    def reset(self) -> None:  # type: ignore[override]
        """Extend parent reset to clear/initialise *opp_actions* buffer."""
        super().reset()
        if not hasattr(self, "opp_actions"):
            # Allocate on first call – ``buffer_size`` and ``n_envs`` are now set.
            import numpy as np  # local import to avoid polluting module scope

            self.opp_actions = np.zeros((self.buffer_size, self.n_envs), dtype=np.int64)
        else:
            self.opp_actions.fill(0)

    # pylint: disable=too-many-arguments
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        opp_action_bucket: np.ndarray,
    ) -> None:  # noqa: D401 – SB3 signature extension
        """Add a transition to the buffer (extra *opp_action_bucket* arg)."""
        super().add(obs, action, reward, episode_start, value, log_prob)
        self.opp_actions[self.pos - 1] = opp_action_bucket  # pos already advanced

    def get(self, batch_size: int = 64):  # noqa: D401 – generator, mirrors parent
        """Yield minibatches along with matching *opp_actions*.

        This implementation delegates the sampling logic to the parent
        :pyfunc:`RolloutBuffer.get` (which returns a generator over
        :class:`RolloutBufferSamples`) **assuming that** it goes through the
        flattened transitions in *sequential* order when ``shuffle`` is set to
        *False* (this is the current behaviour in SB3 ≥ 1.7). We therefore
        temporarily disable shuffling, track a running pointer into the
        flattened buffer, and slice :pyattr:`opp_actions` accordingly.
        """
        # Temporarily disable shuffling to keep ordering deterministic
        shuffle_attr = getattr(self, "_shuffle", None)
        if shuffle_attr is not None:
            orig_shuffle = shuffle_attr
            setattr(self, "_shuffle", False)
        else:
            orig_shuffle = None

        flat_ptr = 0
        flat_opp = self.opp_actions.reshape(-1)
        for rollout_data in super().get(batch_size):
            batch_len = len(rollout_data.advantages)
            opp_slice = flat_opp[flat_ptr : flat_ptr + batch_len]
            flat_ptr += batch_len
            yield rollout_data, th.as_tensor(opp_slice, device=self.device, dtype=th.long)

        # Restore original shuffling behaviour if it exists
        if orig_shuffle is not None:
            setattr(self, "_shuffle", orig_shuffle)


# ---------------------------------------------------------------------------
# IntentPPO algorithm
# ---------------------------------------------------------------------------

class IntentPPO(PPO):
    """PPO with an auxiliary opponent-prediction loss."""

    def __init__(
        self,
        policy: str | type[IntentPPOPolicy] = IntentPPOPolicy,
        env: GymEnv | str | None = None,
        aux_loss_coef: float = 0.1,
        use_balanced_loss: bool = False,
        lambda_schedule: str = "constant",
        lambda_warmup_steps: int = 0,
        total_timesteps: int = 1_000_000,
        **kwargs: Any,
    ) -> None:
        self.aux_loss_coef_max = float(aux_loss_coef)
        self.lambda_schedule = lambda_schedule
        self.lambda_warmup_steps = int(lambda_warmup_steps)
        self.total_timesteps = int(total_timesteps)
        self.aux_loss_coef = 0.0 if lambda_schedule != "constant" else float(aux_loss_coef)
        self.use_balanced_loss = bool(use_balanced_loss)
        # Force use of our custom buffer
        kwargs.setdefault("rollout_buffer_class", RolloutBufferWithOpp)
        super().__init__(policy, env, **kwargs)

    def _compute_lambda(self):
        if self.lambda_schedule == "constant":
            return self.aux_loss_coef_max
        # Progress: 0 to 1 over warmup steps, then hold
        progress = min(self.num_timesteps / max(1, self.lambda_warmup_steps), 1.0)
        if self.lambda_schedule == "cosine":
            # Cosine ramp from 0 to max
            import math
            return self.aux_loss_coef_max * 0.5 * (1 - math.cos(math.pi * progress))
        elif self.lambda_schedule == "linear":
            return self.aux_loss_coef_max * progress
        else:
            return self.aux_loss_coef_max

    # ------------------------------------------------------------------
    # Override collect_rollouts to capture opponent bucket
    # ------------------------------------------------------------------

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):  # type: ignore[override]  # noqa: N803
        from stable_baselines3.common.utils import safe_mean
        assert self._last_obs is not None
        from gymnasium import spaces
        import torch as th
        import numpy as np

        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions_np = actions.cpu().numpy()

            clipped_actions = actions_np
            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    clipped_actions = np.clip(
                        actions_np, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions_np = actions_np.reshape(-1, 1)

            # Extract red-bucket labels from *infos*
            opp_bucket = np.array(
                [info.get("red_bucket", 0) for info in infos], dtype=np.int64
            )

            # Handle time-limit truncation boot-strap
            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None and infos[idx].get(
                    "TimeLimit.truncated", False
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                opp_bucket,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    # ------------------------------------------------------------------
    # Override train() to include auxiliary loss
    # ------------------------------------------------------------------

    def train(self) -> None:  # noqa: D401, C901 – mirrors PPO.train
        # Switch to train mode
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        # Update lambda (aux_loss_coef) based on scheduler
        self.aux_loss_coef = self._compute_lambda()

        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses, pg_losses, value_losses, aux_losses, aux_accs, clip_fractions = [], [], [], [], [], []

        continue_training = True
        for _epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data, opp_batch in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Auxiliary opponent prediction loss
                # Balanced cross-entropy: weight classes inversely proportional to
                # their frequency *in the current minibatch* so rare buckets are
                # not ignored.
                opp_logits = self.policy.opponent_logits(rollout_data.observations)

                if self.use_balanced_loss:
                    num_classes = getattr(self.policy, "opp_num_buckets", int(opp_logits.shape[-1]))
                    counts = th.bincount(opp_batch, minlength=num_classes).float()
                    # Avoid div-by-zero and normalise so average weight ≈ 1
                    weights = (counts.sum() / (counts + 1e-6))
                    weights = weights / weights.mean()
                    aux_ce = F.cross_entropy(opp_logits, opp_batch, weight=weights)
                else:
                    aux_ce = F.cross_entropy(opp_logits, opp_batch)

                aux_losses.append(aux_ce.item())

                # ----- accuracy -----
                with th.no_grad():
                    preds = opp_logits.argmax(dim=-1)
                    acc = (preds == opp_batch).float().mean().item()
                    aux_accs.append(acc)

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.aux_loss_coef * aux_ce
                )

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping due to KL at {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        from stable_baselines3.common.utils import explained_variance, safe_mean
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss", safe_mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", safe_mean(pg_losses))
        self.logger.record("train/value_loss", safe_mean(value_losses))
        self.logger.record("train/aux_ce_loss", safe_mean(aux_losses))
        if aux_accs:
            self.logger.record("train/aux_accuracy", safe_mean(aux_accs))
        self.logger.record("train/clip_fraction", safe_mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/aux_coef", self.aux_loss_coef) 
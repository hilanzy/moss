"""PPO learner."""
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import chex
import distrax
import jax
import jax.numpy as jnp
import rlax

from moss.core import Buffer, Predictor
from moss.learner.base import BaseLearner
from moss.network import Network
from moss.types import Array, LoggingData, Params, StepType, Transition
from moss.utils.loggers import Logger


class PPOLearner(BaseLearner):
  """PPO learner."""

  def __init__(
    self,
    buffer: Buffer,
    predictors: List[Predictor],
    network_maker: Callable[[], Network],
    logger_fn: Callable[..., Logger],
    batch_size: int,
    save_interval: int,
    save_path: str,
    model_path: Optional[str] = None,
    gradient_clip: Optional[float] = None,
    data_reuse: Optional[int] = None,
    publish_interval: int = 1,
    learning_rate: float = 5e-4,
    discount: float = 0.99,
    gae_lambda: float = 0.95,
    pg_clip_epsilon: float = 0.1,
    value_clip_epsilon: Optional[float] = None,
    critic_coef: float = 0.5,
    entropy_coef: float = 0.01,
    seed: int = 42,
  ) -> None:
    """Init."""
    super().__init__(
      buffer, predictors, network_maker, logger_fn, batch_size, save_interval,
      save_path, model_path, gradient_clip, data_reuse, publish_interval,
      learning_rate, seed
    )
    self._discount = discount
    self._gae_lambda = gae_lambda
    self._pg_clip_epsilon = pg_clip_epsilon
    self._value_clip_epsilon = value_clip_epsilon
    self._critic_coef = critic_coef
    self._entropy_coef = entropy_coef

  def _entropy_loss(
    self,
    logits_t: Dict[str, Array],
    w_t: Array,
  ) -> Array:
    """Calculates the entropy regularization loss.

    See "Function Optimization using Connectionist RL Algorithms" by Williams.
    (https://www.tandfonline.com/doi/abs/10.1080/09540099108946587)

    Args:
      logits_t: a sequence of unnormalized action preferences.
      w_t: a per timestep weighting for the loss.

    Returns:
      Entropy loss.
    """
    for logits in logits_t.values():
      chex.assert_rank(logits, 2)
      chex.assert_type(logits, float)
    chex.assert_rank(w_t, 1)
    chex.assert_type(w_t, float)

    distribution = self._network.action_spec.distribution(logits_t)
    entropy_per_timestep = distribution.entropy()
    return -jnp.mean(entropy_per_timestep * w_t)

  def _loss(self, params: Params, data: Transition) -> Tuple[Array, LoggingData]:
    """PPO loss."""
    rnn_state = jax.tree_map(lambda x: x[0], data.rnn_state)
    _, net_output = self._network.forward(
      params, data.state, rnn_state, jax.random.PRNGKey(0), True
    )

    actions, rewards = data.action, data.reward
    behaviour_logits = data.policy_logits
    behaviour_values = data.behaviour_value
    learner_logits, values = net_output.policy_logits, net_output.value
    discount = jnp.ones_like(data.step_type) * self._discount
    # The step is uninteresting if we transitioned LAST -> FIRST.
    mask = jnp.not_equal(data.step_type[:-1], int(StepType.FIRST))
    mask = mask.astype(jnp.float32)

    actions_tm1 = jax.tree_map(lambda x: x[:-1], actions)
    behaviour_logits_tm1 = jax.tree_map(lambda x: x[:-1], behaviour_logits)
    learner_logits_tm1 = jax.tree_map(lambda x: x[:-1], learner_logits)
    values_tm1, behavior_values_tm1 = values[:-1], behaviour_values[:-1]
    rewards_t, discount_t = rewards[1:], discount[1:]

    # Importance sampling.
    rhos = distrax.importance_sampling_ratios(
      self._network.action_spec.distribution(learner_logits_tm1),
      self._network.action_spec.distribution(behaviour_logits_tm1),
      actions_tm1,
    )

    # Computes GAE.
    vmap_generalized_advantage_estimation_fn = jax.vmap(
      rlax.truncated_generalized_advantage_estimation,
      in_axes=[1, 1, None, 1],
      out_axes=1
    )
    advantage_tm1 = vmap_generalized_advantage_estimation_fn(
      rewards_t, discount_t, self._gae_lambda, values
    )

    # Policy gradient loss.
    clipped_surrogate_pg_loss_fn = partial(
      rlax.clipped_surrogate_pg_loss, epsilon=self._pg_clip_epsilon
    )
    vmap_clipped_surrogate_pg_loss_fn = jax.vmap(
      clipped_surrogate_pg_loss_fn, in_axes=1, out_axes=0
    )
    clipped_surrogate_pg_loss = vmap_clipped_surrogate_pg_loss_fn(
      rhos, advantage_tm1
    )
    pg_loss = jnp.mean(clipped_surrogate_pg_loss)

    # Critic loss.
    td_targets = jax.lax.stop_gradient(advantage_tm1 + values_tm1)
    unclipped_td_errors = td_targets - values_tm1
    unclipped_critic_loss = jnp.square(unclipped_td_errors)
    if self._value_clip_epsilon is not None:
      # Clip values to reduce variablility during critic training.
      clipped_values_tm1 = behavior_values_tm1 + jnp.clip(
        values_tm1 - behavior_values_tm1,
        -self._value_clip_epsilon,
        self._value_clip_epsilon,
      )
      clipped_td_errors = td_targets - clipped_values_tm1
      clipped_critic_loss = jnp.square(clipped_td_errors)
      critic_loss = jnp.mean(
        jnp.fmax(unclipped_critic_loss, clipped_critic_loss) * mask
      )
    else:
      critic_loss = jnp.mean(unclipped_critic_loss * mask)
    critic_loss = self._critic_coef * critic_loss

    # Entropy loss.
    vmap_entropy_loss_fn = jax.vmap(self._entropy_loss, in_axes=1, out_axes=0)
    entropy_loss = vmap_entropy_loss_fn(learner_logits_tm1, mask)
    entropy_loss = self._entropy_coef * jnp.mean(entropy_loss)

    # Total loss.
    total_loss = pg_loss + critic_loss + entropy_loss

    # Metrics.
    metrics = {
      "loss/policy": pg_loss,
      "loss/critic": critic_loss,
      "loss/entropy": entropy_loss,
      "loss/total": total_loss,
    }
    return total_loss, metrics

"""PPO learner."""
from functools import partial
from typing import Callable, List, Optional, Tuple

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import rlax

from moss.core import Buffer, Network, Predictor
from moss.learner.base import BaseLearner
from moss.types import (
  Array,
  LoggingData,
  NetOutput,
  Params,
  StepType,
  Transition,
)
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

  def _loss(self, params: Params, data: Transition) -> Tuple[Array, LoggingData]:
    """PPO loss."""
    # Batch forward.
    batch_forward_fn = hk.BatchApply(partial(self._network.forward, params))
    _, net_output = batch_forward_fn(data.state, jax.random.PRNGKey(0))
    net_output: NetOutput

    actions, rewards = data.action, data.reward
    behaviour_logits = data.policy_logits
    behaviour_values = data.behaviour_value
    learner_logits, values = net_output.policy_logits, net_output.value
    discount = jnp.ones_like(data.step_type) * self._discount
    # The step is uninteresting if we transitioned LAST -> FIRST.
    mask = jnp.not_equal(data.step_type[:-1], int(StepType.FIRST))
    mask = mask.astype(jnp.float32)

    actions_tm1, rewards_t = actions[:-1], rewards[1:]
    discount_t = discount[1:]
    behaviour_logits_tm1 = behaviour_logits[:-1]
    learner_logits_tm1 = learner_logits[:-1]
    behavior_values_tm1 = behaviour_values[:-1]
    values_tm1 = values[:-1]

    # Importance sampling.
    rhos = distrax.importance_sampling_ratios(
      distrax.Categorical(learner_logits_tm1),
      distrax.Categorical(behaviour_logits_tm1), actions_tm1
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
    vmap_entropy_loss_fn = jax.vmap(rlax.entropy_loss, in_axes=1, out_axes=0)
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

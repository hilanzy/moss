"""Impala learner."""
from functools import partial
from typing import Callable, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import rlax
from absl import logging

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


class ImpalaLearner(BaseLearner):
  """Impala learner."""

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
    learning_rate: float = 5e-4,
    discount: float = 0.95,
    gae_lambda: float = 0.95,
    seed: int = 42,
  ) -> None:
    """Init."""
    super().__init__(
      buffer, predictors, network_maker, logger_fn, batch_size, save_interval,
      save_path, model_path, learning_rate, seed
    )
    self._discount = discount
    self._gae_lambda = gae_lambda
    logging.info(jax.devices())

  def _loss(self, params: Params, data: Transition) -> Tuple[Array, LoggingData]:
    """Impala loss."""
    # Batch forward.
    batch_forward_fn = hk.BatchApply(partial(self._network.forward, params))
    _, net_output = batch_forward_fn(data.obs, jax.random.PRNGKey(0))
    net_output: NetOutput

    actions, rewards = data.action, data.reward
    behaviour_logits = data.policy_logits
    learner_logits, values = net_output.policy_logits, net_output.value
    disconts = jnp.ones_like(data.step_type) * self._discount
    # The step is uninteresting if we transitioned LAST -> FIRST.
    mask = jnp.not_equal(data.step_type[:-1], int(StepType.FIRST))
    mask = mask.astype(jnp.float32)

    actions_t, rewards_t = actions[1:], rewards[1:]
    disconts_t = disconts[1:]
    behaviour_logits_t = behaviour_logits[1:]
    learner_logits_t = learner_logits[1:]
    values_tm1, values_t = values[:-1], values[1:]

    # Importance sampling.
    rhos = rlax.categorical_importance_sampling_ratios(
      pi_logits_t=learner_logits_t,
      mu_logits_t=behaviour_logits_t,
      a_t=actions_t
    )

    # Critic loss.
    vtrace_td_error_and_advantage_fn = partial(
      rlax.vtrace_td_error_and_advantage, lambda_=self._gae_lambda
    )
    vmap_vtrace_td_error_and_advantage_fn = jax.vmap(
      vtrace_td_error_and_advantage_fn, in_axes=1, out_axes=1
    )
    vtrace_returns = vmap_vtrace_td_error_and_advantage_fn(
      values_tm1, values_t, rewards_t, disconts_t, rhos
    )
    critic_loss = 0.5 * jnp.mean(jnp.square(vtrace_returns.errors) * mask)

    # Policy gradien loss.
    adv_mean = jnp.mean(vtrace_returns.pg_advantage)
    adv_std = jnp.std(vtrace_returns.pg_advantage)
    normal_adv = (vtrace_returns.pg_advantage - adv_mean) / adv_std
    vmap_policy_gradient_loss_fn = jax.vmap(
      rlax.policy_gradient_loss, in_axes=1, out_axes=0
    )
    pg_loss = vmap_policy_gradient_loss_fn(
      learner_logits_t, actions_t, normal_adv, mask
    )
    pg_loss = jnp.mean(pg_loss)

    # Entropy loss.
    vmap_entropy_loss_fn = jax.vmap(rlax.entropy_loss, in_axes=1, out_axes=0)
    entropy_loss = vmap_entropy_loss_fn(learner_logits_t, mask)
    entropy_loss = jnp.mean(entropy_loss)

    # Total loss.
    total_loss = pg_loss + 0.5 * critic_loss + 0.001 * entropy_loss

    # Metrics.
    metrics = {
      "loss/policy": pg_loss,
      "loss/critic": critic_loss,
      "loss/entropy": entropy_loss,
      "loss/total": total_loss,
    }
    return total_loss, metrics

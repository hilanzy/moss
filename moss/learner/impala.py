"""Impala learner."""
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
    gradient_clip: Optional[float] = None,
    data_reuse: Optional[int] = None,
    publish_interval: int = 1,
    learning_rate: float = 5e-4,
    discount: float = 0.99,
    gae_lambda: float = 0.95,
    clip_rho_threshold: float = 1.0,
    clip_pg_rho_threshold: float = 1.0,
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
    self._clip_rho_threshold = clip_rho_threshold
    self._clip_pg_rho_threshold = clip_pg_rho_threshold
    self._critic_coef = critic_coef
    self._entropy_coef = entropy_coef

  def _policy_gradient_loss(
    self,
    logits_t: Dict[str, Array],
    a_t: Dict[str, Array],
    adv_t: Array,
    w_t: Array,
    use_stop_gradient: bool = True,
  ) -> Array:
    """Calculates the policy gradient loss.

    See "Simple Gradient-Following Algorithms for Connectionist RL" by Williams.
    (http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)

    Args:
      a_t: a sequence of actions sampled from the preferences `logits_t`.
      adv_t: the observed or estimated advantages from executing actions `a_t`.
      w_t: a per timestep weighting for the loss.
      use_stop_gradient: bool indicating whether or not to apply stop gradient to
        advantages.

    Returns:
      Loss whose gradient corresponds to a policy gradient update.
    """
    for logits in logits_t.values():
      chex.assert_rank(logits, 2)
      chex.assert_type(logits, float)
    for action in a_t.values():
      chex.assert_rank(action, 1)
      chex.assert_type(action, int)
    chex.assert_rank([adv_t, w_t], [1, 1])
    chex.assert_type([adv_t, w_t], [float, float])

    distribution = self._network.action_spec.distribution(logits_t)
    log_pi_a_t = distribution.log_prob(a_t)
    adv_t = jax.lax.select(
      use_stop_gradient, jax.lax.stop_gradient(adv_t), adv_t
    )
    loss_per_timestep = -log_pi_a_t * adv_t
    return jnp.mean(loss_per_timestep * w_t)

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
    """Impala loss."""
    rnn_state = jax.tree_map(lambda x: x[0], data.rnn_state)
    _, net_output = self._network.forward(
      params, data.state, rnn_state, jax.random.PRNGKey(0), True
    )

    actions, rewards = data.action, data.reward
    behaviour_logits = data.policy_logits
    learner_logits, values = net_output.policy_logits, net_output.value
    discount = jnp.ones_like(data.step_type) * self._discount
    # The step is uninteresting if we transitioned LAST -> FIRST.
    mask = jnp.not_equal(data.step_type[:-1], int(StepType.FIRST))
    mask = mask.astype(jnp.float32)

    actions_tm1 = jax.tree_map(lambda x: x[:-1], actions)
    behaviour_logits_tm1 = jax.tree_map(lambda x: x[:-1], behaviour_logits)
    learner_logits_tm1 = jax.tree_map(lambda x: x[:-1], learner_logits)
    values_tm1, values_t = values[:-1], values[1:]
    rewards_t, discount_t = rewards[1:], discount[1:]

    # Importance sampling.
    rhos = distrax.importance_sampling_ratios(
      self._network.action_spec.distribution(learner_logits_tm1),
      self._network.action_spec.distribution(behaviour_logits_tm1),
      actions_tm1,
    )

    # Critic loss.
    vtrace_td_error_and_advantage_fn = partial(
      rlax.vtrace_td_error_and_advantage,
      lambda_=self._gae_lambda,
      clip_rho_threshold=self._clip_rho_threshold,
      clip_pg_rho_threshold=self._clip_pg_rho_threshold
    )
    vmap_vtrace_td_error_and_advantage_fn = jax.vmap(
      vtrace_td_error_and_advantage_fn, in_axes=1, out_axes=1
    )
    vtrace_returns = vmap_vtrace_td_error_and_advantage_fn(
      values_tm1, values_t, rewards_t, discount_t, rhos
    )
    critic_loss = jnp.mean(jnp.square(vtrace_returns.errors) * mask)
    critic_loss = self._critic_coef * critic_loss

    # Policy gradien loss.
    vmap_policy_gradient_loss_fn = jax.vmap(
      self._policy_gradient_loss, in_axes=1, out_axes=0
    )
    pg_loss = vmap_policy_gradient_loss_fn(
      learner_logits_tm1, actions_tm1, vtrace_returns.pg_advantage, mask
    )
    pg_loss = jnp.mean(pg_loss)

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

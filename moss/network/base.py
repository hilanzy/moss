"""Base network."""
from typing import Any, Tuple

import haiku as hk
import jax.numpy as jnp
import rlax

from moss.core import Network
from moss.network.atari import AtariConv
from moss.types import Array, KeyArray, NetOutput, Observation, Params


class SimpleNet(Network):
  """A simple neural network."""

  def __init__(
    self,
    task: str,
    obs_sepc: Any,
    action_sepc: Any,
  ):
    """Init."""
    super().__init__()
    self._task = task
    self._obs_sepc = obs_sepc
    self._action_sepc = action_sepc
    self._net = hk.without_apply_rng(
      hk.transform(lambda x: AtariConv(action_sepc.num_values)(x))
    )

  def init_params(self, rng: KeyArray) -> Params:
    """Init network's params."""
    dummy_inputs = self._obs_sepc.obs.generate_value()
    dummy_inputs = jnp.expand_dims(dummy_inputs, 0)
    params = self._net.init(rng, dummy_inputs)
    return params

  def forward(self, params: Params, obs: Observation,
              rng: KeyArray) -> Tuple[Array, NetOutput]:
    """Network forward."""
    policy_logits, value = self._net.apply(params, obs)
    action = rlax.softmax().sample(rng, policy_logits)
    return action, NetOutput(policy_logits, value)

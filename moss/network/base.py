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
    obs_spec: Any,
    action_spec: Any,
    use_orthogonal: bool = True,
  ):
    """Init."""
    super().__init__()
    self._obs_spec = obs_spec
    self._action_spec = action_spec
    action_nums = action_spec.num_values
    self._net = hk.without_apply_rng(
      hk.transform(lambda x: AtariConv(action_nums, use_orthogonal)(x))
    )

  def init_params(self, rng: KeyArray) -> Params:
    """Init network's params."""
    dummy_inputs = self._obs_spec.obs.generate_value()
    dummy_inputs = jnp.expand_dims(dummy_inputs, 0)
    params = self._net.init(rng, dummy_inputs)
    return params

  def forward(self, params: Params, obs: Observation,
              rng: KeyArray) -> Tuple[Array, NetOutput]:
    """Network forward."""
    policy_logits, value = self._net.apply(params, obs)
    action = rlax.softmax().sample(rng, policy_logits)
    return action, NetOutput(policy_logits, value)

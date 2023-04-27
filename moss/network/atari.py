"""Base network."""
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from moss.types import AgentState, Array


class AtariDense(hk.Module):
  """A simple dense network."""

  def __init__(self, num_actions: int):
    """Init."""
    super().__init__(name="atari_dense")
    self._num_actions = num_actions

  def __call__(
    self,
    state: AgentState,
  ) -> Tuple[Array, Array]:
    """Process a batch of observations."""
    torso = hk.Sequential(
      [hk.Flatten(),
       hk.Linear(512), jax.nn.relu,
       hk.Linear(256), jax.nn.relu]
    )

    policy_net = hk.Sequential(
      [
        hk.Linear(128), jax.nn.relu,
        hk.Linear(32), jax.nn.relu,
        hk.Linear(self._num_actions)
      ]
    )

    value_net = hk.Sequential(
      [hk.Linear(128), jax.nn.relu,
       hk.Linear(32), jax.nn.relu,
       hk.Linear(1)]
    )

    state = state / 255.
    torso_output = torso(state)
    policy_logits = policy_net(torso_output)
    value = value_net(torso_output)
    value = jnp.squeeze(value, axis=-1)

    return policy_logits, value


class AtariConv(hk.Module):
  """A simple convolution network."""

  def __init__(self, num_actions: int, use_orthogonal: bool = True):
    """Init.

    Args:
      num_actions: Atari game action spaces.
      use_orthogonal: Whether use orthogonal to initialization params weight.
        Following https://arxiv.org/abs/2006.05990, we set orthogonal
        initialization scale factor of 0.01 for last layer of policy network
        and others layers set as default(1.0).
    """
    super().__init__("atari_conv")
    self._num_actions = num_actions
    self._use_orthogonal = use_orthogonal

  def __call__(
    self,
    state: AgentState,
  ) -> Tuple[Array, Array]:
    """Process a batch of observations."""
    w_init = hk.initializers.Orthogonal() if self._use_orthogonal else None
    torso = hk.Sequential(
      [
        hk.Conv2D(
          output_channels=32,
          kernel_shape=8,
          stride=4,
          padding="VALID",
          data_format="NCHW",
          w_init=w_init,
        ), jax.nn.relu,
        hk.Conv2D(
          output_channels=64,
          kernel_shape=4,
          stride=2,
          padding="VALID",
          data_format="NCHW",
          w_init=w_init,
        ), jax.nn.relu,
        hk.Conv2D(
          output_channels=64,
          kernel_shape=3,
          stride=1,
          padding="VALID",
          data_format="NCHW",
          w_init=w_init,
        ), jax.nn.relu,
        hk.Flatten()
      ]
    )

    action_w_init = hk.initializers.Orthogonal(
      scale=0.01
    ) if self._use_orthogonal else None
    policy_net = hk.Sequential(
      [
        hk.Linear(512, w_init=w_init), jax.nn.relu,
        hk.Linear(self._num_actions, w_init=action_w_init)
      ]
    )

    value_net = hk.Sequential(
      [
        hk.Linear(512, w_init=w_init), jax.nn.relu,
        hk.Linear(32, w_init=w_init), jax.nn.relu,
        hk.Linear(1, w_init=w_init)
      ]
    )

    state = state / 255.
    torso_output = torso(state)
    policy_logits = policy_net(torso_output)
    value = value_net(torso_output)
    value = jnp.squeeze(value, axis=-1)

    return policy_logits, value

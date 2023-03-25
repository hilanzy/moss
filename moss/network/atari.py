"""Base network."""
from typing import Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from moss.types import Array, Observation


class AtariDense(hk.Module):
  """A simple dense network."""

  def __init__(self, num_actions: int):
    """Init."""
    super().__init__(name="atari_dense")
    self._num_actions = num_actions

  def __call__(
    self,
    obs: Observation,
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

    obs = obs / 255.
    torso_output = torso(obs)
    policy_logits = policy_net(torso_output)
    value = value_net(torso_output)
    value = jnp.squeeze(value, axis=-1)

    return policy_logits, value


class AtariConv(hk.Module):
  """A simple convolution network."""

  def __init__(self, num_actions: int):
    """Init."""
    super().__init__("atari_conv")
    self._num_actions = num_actions

  def __call__(
    self,
    obs: Observation,
  ) -> Tuple[Array, Array]:
    """Process a batch of observations."""
    torso = hk.Sequential(
      [
        hk.Conv2D(32, 8, 4, padding="VALID", data_format="NCHW"), jax.nn.relu,
        hk.Conv2D(64, 4, 2, padding="VALID", data_format="NCHW"), jax.nn.relu,
        hk.Conv2D(64, 3, 1, padding="VALID", data_format="NCHW"), jax.nn.relu,
        hk.Flatten()
      ]
    )

    policy_net = hk.Sequential(
      [hk.Linear(512), jax.nn.relu,
       hk.Linear(self._num_actions)]
    )

    value_net = hk.Sequential(
      [hk.Linear(512), jax.nn.relu,
       hk.Linear(32), jax.nn.relu,
       hk.Linear(1)]
    )

    obs = obs / 255.
    torso_output = torso(obs)
    policy_logits = policy_net(torso_output)
    value = value_net(torso_output)
    value = jnp.squeeze(value, axis=-1)

    return policy_logits, value

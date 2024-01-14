"""Continuous action."""
from typing import Any, Callable, List, Optional

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from moss.network.action.base import Action
from moss.types import Array, BoundedArray, KeyArray, Numeric


class ContinuousAction(Action):
  """Continuous action."""

  def __init__(
    self,
    name: str,
    hidden_sizes: List[int],
    minimum: Numeric = -1,
    maximum: Numeric = 1,
    use_orthogonal: bool = True
  ) -> None:
    """Continuous action init.

    Args:
      name: Action name.
      hidden_sizes: Hidden sizes of action decoder network.
      minimum: Minimum value of action.
      maximum: Maximum value of action.
      use_orthogonal: Whether use orthogonal to initialization params weight.
        Following https://arxiv.org/abs/2006.05990, we set orthogonal
        initialization scale factor of 0.01 for last layer of policy network
        and others layers set as default(1.0).
    """
    self._name = name
    self._hidden_sizes = hidden_sizes
    self._minimum = minimum
    self._maximum = maximum
    self._spec = BoundedArray((), np.float32, minimum, maximum, name)
    self._use_orthogonal = use_orthogonal

  def decoder(self, inputs: Array, mask: Optional[Array] = None) -> Array:
    """Continuous action decoder."""
    del mask  # NOTE: Continuous action don't support mask.
    kernel_init = nn.initializers.orthogonal() if self._use_orthogonal else None
    action_kernel_init = nn.initializers.orthogonal(
      scale=0.01
    ) if self._use_orthogonal else None

    def mlp() -> Callable:
      """Setup a sequential model form hidden_sizes."""
      layers: List[Any] = []
      for hidden_size in self._hidden_sizes:
        layers.append(nn.Dense(hidden_size, kernel_init=kernel_init))
        layers.append(jax.nn.relu)
      layers.append(nn.Dense(1, kernel_init=action_kernel_init))
      layers.append(jax.nn.tanh)
      return nn.Sequential(layers)

    mu_net = mlp()
    sigma_net = mlp()
    mu = mu_net(inputs)
    sigma = sigma_net(inputs)
    policy_logits = jnp.concatenate([mu, sigma], axis=-1)
    return policy_logits

  def distribution(
    self,
    logists: Array,
  ) -> distrax.ClippedNormal:
    """Action distribution."""
    loc, scale = logists[..., 0], logists[..., 1]
    return distrax.ClippedNormal(loc, scale, self._minimum, self._maximum)

  def sample(self, rng: KeyArray, logists: Array) -> Array:
    """Sample continuous action."""
    distribution = self.distribution(logists)
    return distribution.sample(seed=rng)

  @property
  def spec(self) -> BoundedArray:
    """Get action spec."""
    return self._spec

  @property
  def name(self) -> str:
    """Get action name."""
    return self._name

  @property
  def minimum(self) -> Numeric:
    """Minimum action value."""
    return self._minimum

  @property
  def maximum(self) -> Numeric:
    """Maximun action value."""
    return self._maximum

"""Action decoder."""
from typing import Any, List, Optional, Type

import distrax
import flax.linen as nn
import jax
import numpy as np

from moss.network.action.base import Action
from moss.types import Array, DiscreteArray, KeyArray


class DiscreteAction(Action):
  """Discrete action."""

  def __init__(
    self,
    name: str,
    hidden_sizes: List[int],
    num_actions: int,
    use_orthogonal: bool = True
  ) -> None:
    """Init.

    Args:
      name: Action name.
      hidden_sizes: Hidden sizes of action decoder network.
      num_actions: Discrete action nums.
      use_orthogonal: Whether use orthogonal to initialization params weight.
        Following https://arxiv.org/abs/2006.05990, we set orthogonal
        initialization scale factor of 0.01 for last layer of policy network
        and others layers set as default(1.0).
    """
    self._name = name
    self._hidden_sizes = hidden_sizes
    self._num_actions = num_actions
    self._spec = DiscreteArray(num_actions, dtype=np.int8, name=name)
    self._use_orthogonal = use_orthogonal

  def decoder(self, inputs: Array, mask: Optional[Array] = None) -> Array:
    """Action policy network."""
    init_kwargs, action_init_kwargs = {}, {}
    if self._use_orthogonal:
      init_kwargs["kernel_init"] = nn.initializers.orthogonal()
      action_init_kwargs["kernel_init"] = nn.initializers.orthogonal(scale=0.01)
    layers: List[Any] = []
    for hidden_size in self._hidden_sizes:
      layers.append(nn.Dense(hidden_size, **init_kwargs))
      layers.append(jax.nn.relu)
    layers.append(nn.Dense(self._num_actions, **action_init_kwargs))
    policy_net = nn.Sequential(layers)
    policy_logits = policy_net(inputs)
    if mask is not None:
      policy_logits -= mask * 1e9
    return policy_logits

  def distribution(
    self,
    logits: Array,
    temperature: float = 1.,
    dtype: Type = int
  ) -> distrax.Softmax:
    """Action distribution."""
    return distrax.Softmax(logits, temperature, dtype)

  def sample(
    self,
    rng: KeyArray,
    logits: Array,
    temperature: float = 1.,
    dtype: Type = int
  ) -> Array:
    """Sample discrete action."""
    distribution = self.distribution(logits, temperature, dtype)
    return distribution.sample(seed=rng)

  @property
  def spec(self) -> DiscreteArray:
    """Get action spec."""
    return self._spec

  @property
  def name(self) -> str:
    """Get action name."""
    return self._name

  @property
  def num_actions(self) -> int:
    """Get discrete action nums."""
    return self._num_actions

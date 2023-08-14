"""Action decoder."""
from typing import Any, List, Type

import distrax
import haiku as hk
import jax
import numpy as np
from dm_env.specs import Array as ArraySpec

from moss.network.action.base import Action
from moss.types import Array, KeyArray


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
    self._spec = ArraySpec((num_actions,), dtype=np.int8, name=name)
    self._use_orthogonal = use_orthogonal

  def decoder_net(self, inputs: Array) -> Array:
    """Decoder network."""
    w_init = hk.initializers.Orthogonal() if self._use_orthogonal else None
    action_w_init = hk.initializers.Orthogonal(
      scale=0.01
    ) if self._use_orthogonal else None
    layers: List[Any] = []
    for hidden_size in self._hidden_sizes:
      layers.append(hk.Linear(hidden_size, w_init=w_init))
      layers.append(jax.nn.relu)
    layers.append(hk.Linear(self._num_actions, w_init=action_w_init))
    policy_net = hk.Sequential(layers)
    policy_logits = policy_net(inputs)
    return policy_logits

  @classmethod
  def distribution(
    cls,
    logits: Array,
    temperature: float = 1.,
    dtype: Type = int
  ) -> distrax.Softmax:
    """Action distribution."""
    return distrax.Softmax(logits, temperature, dtype)

  @classmethod
  def sample(
    cls,
    rng: KeyArray,
    logits: Array,
    temperature: float = 1.,
    dtype: Type = int
  ) -> Array:
    """Sample discrete action."""
    distribution = cls.distribution(logits, temperature, dtype)
    return distribution.sample(seed=rng)

  @property
  def num_actions(self) -> int:
    """Get discrete action nums."""
    return self._num_actions

  @property
  def spec(self) -> ArraySpec:
    """Get action spec."""
    return self._spec

"""Action decoder."""
from typing import Any, Type

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
    self, name: str, num_actions: int, use_orthogonal: bool = True
  ) -> None:
    """Init.

    Args:
      name: Action name.
      num_actions: Discrete action nums.
      use_orthogonal: Whether use orthogonal to initialization params weight.
        Following https://arxiv.org/abs/2006.05990, we set orthogonal
        initialization scale factor of 0.01 for last layer of policy network
        and others layers set as default(1.0).
    """
    self._name = name
    self._num_actions = num_actions
    self._spec = ArraySpec((num_actions,), dtype=np.int8, name=name)
    self._use_orthogonal = use_orthogonal

  def decoder_net(self, inputs: Any) -> Any:
    """Decoder network."""
    w_init = hk.initializers.Orthogonal() if self._use_orthogonal else None
    action_w_init = hk.initializers.Orthogonal(
      scale=0.01
    ) if self._use_orthogonal else None
    policy_net = hk.Sequential(
      [
        hk.Linear(512, w_init=w_init), jax.nn.relu,
        hk.Linear(self._num_actions, w_init=action_w_init)
      ]
    )
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

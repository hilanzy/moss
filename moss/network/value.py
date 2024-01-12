"""Value decoder network."""
from typing import Any, List

import haiku as hk
import jax
import jax.numpy as jnp

from moss.network.base import Module


class DenseValue(Module):
  """Dense value network."""

  def __init__(
    self, name: str, hidden_sizes: List[int], use_orthogonal: bool = True
  ):
    """Init."""
    super().__init__(name)
    self._hidden_sizes = hidden_sizes
    self._use_orthogonal = use_orthogonal

  def __call__(self, inputs: Any) -> Any:
    """Call."""
    w_init = hk.initializers.Orthogonal() if self._use_orthogonal else None
    mlp_layers: List[Any] = []
    for hidden_size in self._hidden_sizes:
      mlp_layers.append(hk.Linear(hidden_size, w_init=w_init))
      mlp_layers.append(jax.nn.relu)
    mlp_layers.append(hk.Linear(1, w_init=w_init))
    value_net = hk.Sequential(mlp_layers)
    value = value_net(inputs)
    value = jnp.squeeze(value, axis=-1)
    return value

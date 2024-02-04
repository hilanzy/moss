"""Value decoder network."""
from typing import Any, List, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from moss.types import Array


class DenseValue(object):
  """Dense value decoder."""

  def __init__(
    self,
    hidden_sizes: List[int],
    name: Optional[str] = None,
    use_orthogonal: bool = True,
  ) -> None:
    """Dense value init."""
    self._name = name or "dense_value"
    self._hidden_sizes = hidden_sizes
    self._use_orthogonal = use_orthogonal

  def decoder(self, inputs: Array) -> Array:
    """Value decoder."""
    layers: List[Any] = []
    init_kwargs = {}
    if self._use_orthogonal:
      init_kwargs["kernel_init"] = nn.initializers.orthogonal()
    for hidden_size in self._hidden_sizes:
      layers.append(nn.Dense(hidden_size, **init_kwargs))
      layers.append(jax.nn.relu)
    layers.append(nn.Dense(1, **init_kwargs))
    value_net = nn.Sequential(layers)
    value = value_net(inputs)
    value = jnp.squeeze(value, axis=-1)
    return value

  def name(self) -> str:
    """Get value name."""
    return self._name

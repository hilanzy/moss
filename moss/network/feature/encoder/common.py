"""Common feature encoder."""
from typing import Any, List

import haiku as hk
import jax

from moss.types import Array


class CommonEncoder(hk.Module):
  """Common encoder."""

  def __init__(
    self,
    name: str,
    hidden_sizes: List[int],
    use_orthogonal: bool = True
  ) -> None:
    """Init."""
    super().__init__(name)
    self._hidden_sizes = hidden_sizes
    self._use_orthogonal = use_orthogonal

  def __call__(self, inputs: Array) -> Any:
    """Call."""
    layers: List[Any] = []
    w_init = hk.initializers.Orthogonal() if self._use_orthogonal else None
    for hidden_size in self._hidden_sizes:
      layers.append(hk.Linear(hidden_size, w_init=w_init))
      layers.append(jax.nn.relu)
    common_net = hk.Sequential(layers)
    encoder_out = common_net(inputs)
    return encoder_out

"""Common feature encoder."""
from typing import Any, List

import flax.linen as nn
import jax

from moss.network.feature.encoder.base import FeatureEncoder
from moss.types import Array


class CommonEncoder(FeatureEncoder):
  """Common encoder."""

  def __init__(
    self,
    name: str,
    hidden_sizes: List[int],
    use_orthogonal: bool = True
  ) -> None:
    """Init."""
    self._name = name
    self._hidden_sizes = hidden_sizes
    self._use_orthogonal = use_orthogonal

  def __call__(self, inputs: Array) -> Any:
    """Call."""
    layers: List[Any] = []
    init_kwargs = {}
    if self._use_orthogonal:
      init_kwargs["kernel_init"] = nn.initializers.orthogonal()
    for hidden_size in self._hidden_sizes:
      layers.append(nn.Dense(hidden_size, **init_kwargs))
      layers.append(jax.nn.relu)
    common_net = nn.Sequential(layers)
    encoder_out = common_net(inputs)
    return encoder_out

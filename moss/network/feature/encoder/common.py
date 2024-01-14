"""Common feature encoder."""
from typing import Any, List

import flax.linen as nn
import jax

from moss.types import Array


class CommonEncoder(nn.Module):
  """Common encoder."""
  hidden_sizes: List[int]
  use_orthogonal: bool = True

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Call."""
    layers: List[Any] = []
    kernel_init = nn.initializers.orthogonal() if self.use_orthogonal else None
    for hidden_size in self.hidden_sizes:
      layers.append(nn.Dense(hidden_size, kernel_init=kernel_init))
      layers.append(jax.nn.relu)
    common_net = nn.Sequential(layers)
    encoder_out = common_net(inputs)
    return encoder_out

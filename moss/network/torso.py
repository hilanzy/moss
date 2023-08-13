"""Torso decoder network."""
from typing import Any, Dict, List

import haiku as hk
import jax
import jax.numpy as jnp


class DenseTorso(hk.Module):
  """Dense torso network."""

  def __init__(
    self, name: str, hidden_sizes: List[int], use_orthogonal: bool = True
  ):
    """Init."""
    super().__init__(name)
    self._hidden_sizes = hidden_sizes
    self._use_orthogonal = use_orthogonal

  def __call__(self, inputs: Dict[str, Any]) -> Any:
    """Call."""
    mlp_layers: List[Any] = []
    w_init = hk.initializers.Orthogonal() if self._use_orthogonal else None
    for hidden_size in self._hidden_sizes:
      mlp_layers.append(hk.Linear(hidden_size, w_init=w_init))
      mlp_layers.append(jax.nn.relu)
    torso_net = hk.Sequential(mlp_layers)
    torso_input = jnp.concatenate(list(inputs.values()), axis=0)
    torso_out = torso_net(torso_input)
    return torso_out

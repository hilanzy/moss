"""Dense torso network."""
from typing import Any, Dict, List, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from moss.network.base import Module
from moss.types import Array


class DenseTorso(Module):
  """Dense torso network."""

  def __init__(
    self, name: str, hidden_sizes: List[int], use_orthogonal: bool = True
  ):
    """Init."""
    super().__init__(name)
    self._hidden_sizes = hidden_sizes
    self._use_orthogonal = use_orthogonal

  def __call__(self, inputs: Dict[str, Any],
               rnn_state: Any) -> Tuple[Array, Any]:
    """Dense torso network forward.

    Args:
      inputs: input features.
      rnn_state: rnn state, ignore for this class.
    """
    mlp_layers: List[Any] = []
    w_init = hk.initializers.Orthogonal() if self._use_orthogonal else None
    for hidden_size in self._hidden_sizes:
      mlp_layers.append(hk.Linear(hidden_size, w_init=w_init))
      mlp_layers.append(jax.nn.relu)
    torso_net = hk.Sequential(mlp_layers)
    torso_input = jnp.concatenate(list(inputs.values()), axis=-1)
    torso_out = torso_net(torso_input)
    return torso_out, rnn_state

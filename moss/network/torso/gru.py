"""GRU torso network."""
from typing import Any, Dict, Optional, Tuple

import haiku as hk
import jax.numpy as jnp

from moss.types import Array, RNNState


class GRUTorso(hk.RNNCore):
  """GRU torso network."""

  def __init__(self, name: str, hidden_size: int, use_orthogonal: bool = True):
    """Init."""
    super().__init__(name)
    w_init = hk.initializers.Orthogonal() if use_orthogonal else None
    self._core = hk.GRU(hidden_size, w_i_init=w_init, w_h_init=w_init)

  def __call__(self, inputs: Dict[str, Any],
               state: RNNState) -> Tuple[Array, RNNState]:
    """Call."""
    torso_input = jnp.concatenate(list(inputs.values()), axis=-1)
    torso_out, new_state = self._core(torso_input, state)
    return torso_out, new_state

  def initial_state(self, batch_size: Optional[int]) -> RNNState:
    """Constructs an initial state for this core."""
    return self._core.initial_state(batch_size)

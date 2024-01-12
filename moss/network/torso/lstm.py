"""LSTM torso network."""
from typing import Any, Dict, Optional, Tuple

import haiku as hk
import jax.numpy as jnp

from moss.network.base import RNNModule
from moss.types import Array


class LSTMTorso(RNNModule):
  """LSTM torso network."""

  def __init__(self, name: str, hidden_size: int):
    """Init."""
    super().__init__(name)
    self._core = hk.LSTM(hidden_size)

  def __call__(self, inputs: Dict[str, Any],
               state: hk.LSTMState) -> Tuple[Array, hk.LSTMState]:
    """Call."""
    torso_input = jnp.concatenate(list(inputs.values()), axis=-1)
    torso_out, new_state = self._core(torso_input, state)
    return torso_out, new_state

  def initial_state(self, batch_size: Optional[int]) -> hk.LSTMState:
    """Constructs an initial state for this core."""
    return self._core.initial_state(batch_size)

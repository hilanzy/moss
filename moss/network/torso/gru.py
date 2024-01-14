"""GRU torso network."""
from typing import Optional, Tuple

import flax.linen as nn

from moss.network.torso.base import RNNTorso
from moss.types import Array, RNNState


class GRUTorso(RNNTorso):
  """GRU torso network."""

  def __init__(
    self,
    hidden_size: int,
    use_orthogonal: bool = True,
    name: Optional[str] = None
  ) -> None:
    """GRU torso init."""
    self._name = name or "gru_torso"
    self._hidden_size = hidden_size
    self._use_orthogonal = use_orthogonal

  def aggregator(
    self,
    inputs: Array,
    rnn_state: RNNState,
    training: bool = False
  ) -> Tuple[Array, RNNState]:
    """GRU torso aggregator.

    Args:
      inputs: input features.
      rnn_state: gru rnn state.

    Returns:
      torso_out: torso aggregator output.
      new_state: new gru rnn state.
    """
    kernel_init = nn.initializers.orthogonal() if self._use_orthogonal else None
    gru_cell = nn.GRUCell(
      self._hidden_size,
      kernel_init=kernel_init,
      recurrent_kernel_init=kernel_init
    )
    if training:
      new_state, torso_out = nn.RNN(
        gru_cell, return_carry=True, time_major=True
      )(inputs, initial_carry=rnn_state)  # type: ignore
    else:
      if rnn_state is None:
        rnn_state = self.initial_state(gru_cell, inputs)
      new_state, torso_out = gru_cell(rnn_state, inputs)
    return torso_out, new_state

  @property
  def name(self) -> str:
    """Get torso name."""
    return self._name

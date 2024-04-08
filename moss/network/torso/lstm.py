"""LSTM torso network."""
from typing import Optional, Tuple

import flax.linen as nn

from moss.network.torso.base import RNNTorso
from moss.types import Array, RNNState


class LSTMTorso(RNNTorso):
  """LSTM torso network."""

  def __init__(
    self,
    hidden_size: int,
    use_orthogonal: bool = True,
    name: Optional[str] = None
  ) -> None:
    """LSTM torso init."""
    self._name = name or "lstm_torso"
    self._hidden_size = hidden_size
    self._use_orthogonal = use_orthogonal

  def aggregator(
    self,
    inputs: Array,
    rnn_state: RNNState,
    training: bool = False
  ) -> Tuple[Array, RNNState]:
    """LSTM torso aggregator.

    Args:
      inputs: input features.
      rnn_state: lstm rnn state.

    Returns:
      torso_out: torso aggregator output.
      new_state: new lstm rnn state.
    """
    init_kwargs = {}
    if self._use_orthogonal:
      init_kwargs["kernel_init"] = nn.initializers.orthogonal()
      init_kwargs["recurrent_kernel_init"] = nn.initializers.orthogonal()
    lstm_cell = nn.OptimizedLSTMCell(
      self._hidden_size,
      **init_kwargs,
    )
    if training:
      new_state, torso_out = nn.RNN(
        lstm_cell, time_major=True, return_carry=True
      )(inputs, initial_carry=rnn_state)  # type: ignore
    else:
      if rnn_state is None:
        rnn_state = self.initial_state(lstm_cell, inputs)
      new_state, torso_out = lstm_cell(rnn_state, inputs)  # type: ignore
    return torso_out, new_state

  @property
  def name(self) -> str:
    """Get torso name."""
    return self._name

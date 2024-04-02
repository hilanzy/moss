"""Torso base."""
import abc
from typing import Tuple

import flax.linen as nn
import jax

from moss.types import Array, KeyArray, RNNState


class Torso(abc.ABC):
  """Torso."""

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Torso name."""

  @abc.abstractmethod
  def aggregator(
    self,
    inputs: Array,
    rnn_state: RNNState,
    training: bool = False
  ) -> Tuple[Array, RNNState]:
    """Aggregator.

    Args:
      inputs: input features.
      rnn_state: rnn state.
      training: whether is trainig mode.

    Returns:
      torso_out: torso aggregator output.
      new_state: new rnn state.
    """


class RNNTorso(Torso):
  """RNN torso."""

  def initial_state(
    self,
    cell: nn.RNNCellBase,
    inputs: Array,
    time_major: bool = True,
    rng: KeyArray = None
  ) -> RNNState:
    """Constructs an initial state for rnn core.

    Args:
      cell: RNN cell.
      inputs: the inputs for RNN cell.
      time_major: if `time_major=False` it will expect inputs with shape
        `(*batch, time, *features)`, else it will expect inputs with shape
        `(time, *batch, *features)`.
      rng: a PRNG key used to initialize the carry, if not provided
        `jax.random.PRNGKey(0)` will be used. Most cells will ignore this

    Returns:
      An initialized rnn state for the given RNN cell.
    """
    if rng is None:
      rng = jax.random.PRNGKey(0)
    time_axis = (0 if time_major else inputs.ndim - (cell.num_feature_axes + 1))
    input_shape = inputs.shape[:time_axis] + inputs.shape[time_axis + 1:]
    rnn_state = cell.initialize_carry(rng, input_shape)
    return rnn_state

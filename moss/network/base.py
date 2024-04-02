"""Base network."""
import abc
from typing import Dict, Tuple

from moss.network.action import ActionSpec
from moss.types import Array, KeyArray, NetOutput, Params, RNNState


class Network(abc.ABC):
  """Neural network interface."""

  @property
  @abc.abstractmethod
  def action_spec(self) -> ActionSpec:
    """Action spec."""

  @abc.abstractmethod
  def initial_params(self, rng: KeyArray) -> Params:
    """Init network's params."""

  @abc.abstractmethod
  def forward(
    self, params: Params, input_dict: Dict, rnn_state: RNNState, rng: KeyArray,
    training: bool
  ) -> Tuple[Dict[str, Array], NetOutput]:
    """Network forward."""

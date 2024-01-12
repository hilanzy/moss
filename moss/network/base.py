"""Base network."""
import abc
from typing import Any, Dict, Optional, Tuple

from moss.network.action import ActionSpec
from moss.types import Array, KeyArray, NetOutput, Params, RNNState


class Module(abc.ABC):
  """Net module."""

  def __init__(self, name: Optional[str] = None) -> None:
    """Init."""
    self._name = name

  @abc.abstractmethod
  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Module forward."""


class RNNModule(Module):
  """RNN module."""

  @abc.abstractmethod
  def initial_state(self, batch_size: Optional[int]) -> RNNState:
    """Initial rnn state."""


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
  def initial_state(self, batch_size: Optional[int]) -> Any:
    """Constructs an initial state for rnn core."""

  @abc.abstractmethod
  def forward(
    self, params: Params, input_dict: Dict, rnn_state: RNNState, rng: KeyArray,
    training: bool
  ) -> Tuple[Dict[str, Array], NetOutput]:
    """Network forward."""

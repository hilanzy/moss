"""Core interface."""
import abc
from typing import Any, Dict, Tuple

from moss.env.base import TimeStep
from moss.types import (
  Array,
  LoggingData,
  OptState,
  Params,
  Reward,
  RNNState,
  Trajectory,
  Transition,
)


class Worker(abc.ABC):
  """Worker interface."""

  @abc.abstractmethod
  def run(self) -> None:
    """Runs the worker."""


class Actor(Worker):
  """Actor interface."""


class Agent(abc.ABC):
  """Agent interface."""

  @abc.abstractmethod
  def reset(self) -> Any:
    """Reset agent."""

  @abc.abstractmethod
  def step(self, timestep: TimeStep) -> Tuple[Dict, Reward]:
    """Trans env timestep into network input dict.

    Args:
      timestep: Env timestep.

    Returns:
      input_dict: The input dict for network.
        input keys:
          `AGENT_STATE`: the agent state for network.
          `GLOBAL_STATE`: the global state for network(CTDE).
          `MASK`: the action mask.
    """

  @abc.abstractmethod
  def inference(self, input_dict: Dict) -> Any:
    """Inference."""

  @abc.abstractmethod
  def result(self, idx: int) -> Any:
    """Get inference result async."""

  @abc.abstractmethod
  def take_action(self, action: Dict[str, Any]) -> Any:
    """Take action."""

  @abc.abstractmethod
  def add(self, transition: Transition) -> None:
    """Add transition to buffer."""


class Buffer(abc.ABC):
  """Replay buffer interface."""

  @abc.abstractmethod
  def add(self, traj: Trajectory) -> None:
    """Add replay transtion."""

  @abc.abstractmethod
  def sample(self, sample_size: int) -> Transition:
    """Sample data."""


class Learner(Worker):
  """RL learner interface."""

  @abc.abstractmethod
  def _loss(self, params: Params, data: Transition) -> Tuple[Array, LoggingData]:
    """Loss function."""

  @abc.abstractmethod
  def _train_step(
    self, params: Params, opt_state: OptState, data: Transition
  ) -> Any:
    """Training step."""

  @abc.abstractmethod
  def _publish_params(self, params: Params) -> Any:
    """Publish params to pedictors."""

  @abc.abstractmethod
  def _load_model(self, model_path: str) -> Any:
    """Load model."""

  @abc.abstractmethod
  def _save_model(self, save_path: str, params: Params) -> None:
    """Save model."""


class Predictor(Worker):
  """Predictor interface."""

  @abc.abstractmethod
  def update_params(self, params: Params) -> None:
    """Update params."""

  @abc.abstractmethod
  def inference(self, input_dict: Dict, rnn_state: RNNState) -> Any:
    """Inference."""

  @abc.abstractmethod
  def result(self, idx: int) -> Any:
    """Get result async."""

"""Core interface."""
import abc
from typing import Any, Tuple

from moss.env.base import TimeStep
from moss.types import (
  AgentState,
  Array,
  KeyArray,
  LoggingData,
  NetOutput,
  OptState,
  Params,
  Reward,
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
  def step(self, timestep: TimeStep) -> Tuple[AgentState, Reward]:
    """Take step."""

  @abc.abstractmethod
  def take_action(self, state: AgentState) -> Any:
    """Take action."""


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


class Network(abc.ABC):
  """Neural network interface."""

  @abc.abstractmethod
  def init_params(self, rng: KeyArray) -> Params:
    """Init network's params."""

  @abc.abstractmethod
  def forward(self, params: Params, state: AgentState,
              rng: KeyArray) -> Tuple[Array, NetOutput]:
    """Network forward."""


class Predictor(Worker):
  """Predictor interface."""

  @abc.abstractmethod
  def update_params(self, params: Params) -> None:
    """Update params."""

  @abc.abstractmethod
  def inference(self, state: AgentState) -> Any:
    """Inference."""

  @abc.abstractmethod
  def result(self, idx: int) -> Any:
    """Get result async."""

"""Base environment worker."""
import abc
from typing import Any

from moss.env import BaseEnv


class BaseEnvWorker(abc.ABC):
  """Base environment worker."""

  def __init__(self, env: BaseEnv) -> None:
    """Init."""
    self._env = env

  @abc.abstractmethod
  def reset(self) -> Any:
    """Reset."""

  @abc.abstractmethod
  def step(self, actions: Any) -> Any:
    """Step."""


class DummyWorker(BaseEnvWorker):
  """Dummy environment worker."""

  def __init__(self, env: BaseEnv) -> None:
    """Init."""
    super().__init__(env)

  def reset(self) -> Any:
    """Dummy worker reset."""
    return self._env.reset()

  def step(self, actions: Any) -> Any:
    """Dummy worker step."""
    return self._env.step(actions)

"""Base environment."""
import abc
from typing import Any

from moss.types import Environment


class BaseEnv(Environment):
  """Abstract base environments class for moss."""

  @abc.abstractmethod
  def reset(self) -> Any:
    """Starts a new environment."""

  @abc.abstractmethod
  def step(self, action: Any) -> Any:
    """Updates the environment."""

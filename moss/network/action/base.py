"""Base action."""
import abc
from typing import Any, Optional

from distrax import DistributionLike

from moss.types import Array, SpecArray


class Action(abc.ABC):
  """Action."""

  @abc.abstractmethod
  def decoder(self, inputs: Array, mask: Optional[Array] = None) -> Array:
    """Action decoder."""

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Action name."""

  @property
  @abc.abstractmethod
  def spec(self) -> SpecArray:
    """Action spec."""

  @abc.abstractmethod
  def distribution(self, *args: Any, **kwargs: Any) -> DistributionLike:
    """Action distribution."""

  @abc.abstractmethod
  def sample(self, *args: Any, **kwargs: Any) -> Any:
    """Sample action."""

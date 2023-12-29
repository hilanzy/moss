"""Base action."""
import abc
from typing import Any, Optional

from distrax import DistributionLike

from moss.types import Array, SpecArray


class Action(abc.ABC):
  """Action."""

  @property
  @abc.abstractmethod
  def name(self) -> SpecArray:
    """Get action name."""

  @property
  @abc.abstractmethod
  def spec(self) -> SpecArray:
    """Action spec."""

  @abc.abstractmethod
  def policy_net(self, inputs: Array, mask: Optional[Array] = None) -> Array:
    """Action policy network."""

  @classmethod
  @abc.abstractmethod
  def distribution(cls, *args: Any, **kwargs: Any) -> DistributionLike:
    """Action distribution."""

  @classmethod
  @abc.abstractmethod
  def sample(cls, *args: Any, **kwargs: Any) -> Any:
    """Sample action."""

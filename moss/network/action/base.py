"""Base action."""
import abc
from typing import Any, Optional

from distrax import DistributionLike
from dm_env.specs import Array as ArraySpec

from moss.types import Array


class Action(abc.ABC):
  """Action."""

  @property
  @abc.abstractmethod
  def mask_on(self) -> ArraySpec:
    """Action mask on."""

  @property
  @abc.abstractmethod
  def spec(self) -> ArraySpec:
    """Action spec."""

  @abc.abstractmethod
  def policy_net(self, inputs: Any, mask: Optional[Array] = None) -> Any:
    """Action policy network."""

  @classmethod
  @abc.abstractmethod
  def distribution(cls, *args: Any, **kwargs: Any) -> DistributionLike:
    """Action distribution."""

  @classmethod
  @abc.abstractmethod
  def sample(cls, *args: Any, **kwargs: Any) -> Any:
    """Sample action."""

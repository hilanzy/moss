"""Base action."""
import abc
from typing import Any

from distrax import DistributionLike
from dm_env.specs import Array as ArraySpec


class Action(abc.ABC):
  """Action."""

  @property
  @abc.abstractmethod
  def spec(self) -> ArraySpec:
    """Action spec."""

  @abc.abstractmethod
  def decoder_net(self, inputs: Any) -> Any:
    """Decoder network."""

  @classmethod
  @abc.abstractmethod
  def distribution(cls, *args: Any, **kwargs: Any) -> DistributionLike:
    """Action distribution."""

  @classmethod
  @abc.abstractmethod
  def sample(cls, *args: Any, **kwargs: Any) -> Any:
    """Sample action."""

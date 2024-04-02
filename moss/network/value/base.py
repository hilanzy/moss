"""Value decoder head."""
import abc

from moss.types import Array


class Value(abc.ABC):
  """Base value decoder."""

  @abc.abstractmethod
  def decoder(self, inputs: Array) -> Array:
    """Value decoder."""

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Get value name."""

"""Base feature encoder."""
import abc
from typing import Any


class FeatureEncoder(abc.ABC):
  """Feature encoder."""

  @abc.abstractmethod
  def __call__(self, inputs: Any) -> Any:
    """Feature encoder function."""

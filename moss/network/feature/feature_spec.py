"""Feature spec."""
from typing import Any, Dict

from moss.network.feature.feature_set import FeatureSet
from moss.types import SpecArray


class FeatureSpec(object):
  """Feature spec."""

  def __init__(self, feature_sets: Dict[str, FeatureSet]) -> None:
    """Init."""
    self._feature_sets = feature_sets

  @property
  def feature_sets(self) -> Dict[str, FeatureSet]:
    """Feature sets."""
    return self._feature_sets

  @property
  def spec(self) -> Dict[str, Dict[str, SpecArray]]:
    """Feature spec."""
    feature_spec = {
      name: feature_set.spec for name, feature_set in self._feature_sets.items()
    }
    return feature_spec

  def generate_value(self) -> Dict[str, Any]:
    """Generate a test value which conforms to this feature spec."""
    value = {
      name: feature_set.generate_value()
      for name, feature_set in self._feature_sets.items()
    }
    return value

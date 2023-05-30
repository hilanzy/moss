"""Feature set."""
from typing import Any, Callable, Dict

import haiku as hk
import jax.numpy as jnp
import tree

from moss.network.feature import BaseFeature
from moss.types import Array


class CommonFeatureSet(object):
  """Common feature set."""

  def __init__(
    self, name: str, features: Dict[str, BaseFeature],
    encoder_net_maker: Callable[[], hk.Module]
  ) -> None:
    """Init."""
    self._name = name
    self._features = features
    self._encoder_net_maker = encoder_net_maker

  @property
  def name(self) -> str:
    """Get name."""
    return self._name

  @property
  def spec(self) -> Dict[str, BaseFeature]:
    """Get features spec dict."""
    return self._features

  @property
  def encoder_net_maker(self) -> Callable[[], Any]:
    """Feature encoder."""
    return self._encoder_net_maker

  def generate_value(self) -> Any:
    """Generate a test value which conforms to this feature set."""
    return tree.map_structure(lambda spec: spec.generate_value(), self.spec)

  def process(self, features: Dict) -> Array:
    """Features process."""
    feature_list = []
    for feature in features.values():
      feature_list.append(feature)
    output = jnp.concatenate(feature_list, axis=-1)
    return output

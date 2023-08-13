"""Feature set."""
from typing import Any, Callable, Dict

import haiku as hk
import jax.numpy as jnp
import tree

from moss.network.feature.feature import BaseFeature
from moss.types import Array


class FeatureSet(object):
  """Feature set."""

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

  def process(self, feature_inputs: Dict) -> Array:
    """Feature inputs process."""
    output_list = []
    for name, inputs in feature_inputs.items():
      feature = self._features.get(name)
      if feature is None:
        raise ValueError(
          f"Got unexpected feature name `{name}` in `{self.name}` feature set."
        )
      output_list.append(feature.process(inputs))
    output = jnp.concatenate(output_list, axis=0)
    return output

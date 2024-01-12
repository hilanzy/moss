"""Feature set."""
from typing import Any, Dict

import jax.numpy as jnp
import tree

from moss.network.base import Module
from moss.network.feature.feature import BaseFeature
from moss.types import Array


class FeatureSet(object):
  """Feature set."""

  def __init__(
    self, name: str, features: Dict[str, BaseFeature], encoder_net: Module
  ) -> None:
    """Init."""
    self._name = name
    self._features = features
    self._encoder_net = encoder_net

  @property
  def name(self) -> str:
    """Get name."""
    return self._name

  @property
  def spec(self) -> Dict[str, BaseFeature]:
    """Get features spec dict."""
    return self._features

  @property
  def encoder_net(self) -> Module:
    """Feature encoder."""
    return self._encoder_net

  def generate_value(self) -> Any:
    """Generate a test value which conforms to this feature set."""
    return tree.map_structure(lambda spec: spec.generate_value(), self.spec)

  def process(self, feature_inputs: Dict) -> Array:
    """Feature inputs process."""
    output_list = []
    for name, feature in self._features.items():
      inputs = feature_inputs.get(name)
      if inputs is None:
        raise ValueError(
          f"Not find the inputs named `{name}` in `{self.name}` feature set, "
          "please check you inputs."
        )
      output_list.append(feature.process(inputs))
    output = jnp.concatenate(output_list, axis=0)
    return output

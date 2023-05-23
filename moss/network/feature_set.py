"""Feature set."""
from typing import Any, Dict

import jax.numpy as jnp


class CommonFeatureSet(object):
  """Common feature set."""

  def __init__(self, features: Dict, encoder: Any) -> None:
    """Init."""
    self._features = features
    self._encoder = encoder

  def process(self, features: Dict) -> Any:
    """Features process."""
    feature_list = []
    for feature in features.values():
      feature_list.append(feature)
    output = jnp.concatenate(feature_list, axis=-1)
    return output

"""Feature."""
from typing import Any, Optional, Tuple

import jax.numpy as jnp
from dm_env.specs import Array


class BaseFeature(Array):
  """Base feature."""

  def __init__(
    self, shape: Tuple, dtype: Any, name: Optional[str] = None
  ) -> None:
    """Init."""
    super().__init__(shape=shape, dtype=dtype, name=name)

  def process(self, inputs: Any) -> Any:
    """Feature process."""
    return jnp.array(inputs)

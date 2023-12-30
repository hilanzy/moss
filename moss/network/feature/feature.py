"""Feature."""
from typing import Any, Callable, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np

from moss.types import SpecArray


class BaseFeature(SpecArray):
  """Base feature."""

  def __init__(
    self,
    shape: Tuple,
    dtype: Type = np.float32,
    name: Optional[str] = None,
    preprocess_fn: Optional[Callable[..., Any]] = None
  ) -> None:
    """Init."""
    super().__init__(shape, dtype, name)
    self._preprocess_fn = preprocess_fn

  def process(self, inputs: Any) -> Any:
    """Feature process."""
    if self._preprocess_fn is not None:
      inputs = self._preprocess_fn(inputs)
    feature = jnp.array(inputs)
    return feature


class ScalarFeature(BaseFeature):
  """Scalar feature."""

  def __init__(
    self,
    dtype: Type = np.float32,
    name: Optional[str] = "ScalarFeature",
    preprocess_fn: Optional[Callable[..., Any]] = None
  ) -> None:
    """Init."""
    super().__init__((1,), dtype, name, preprocess_fn)

  def process(self, inputs: Any) -> Any:
    """Feature process."""
    if self._process_fn is not None:
      inputs = self._process_fn(inputs)
    feature = jnp.array([inputs])
    return feature


class VectorFeature(BaseFeature):
  """Vector feature."""

  def __init__(
    self,
    length: int,
    dtype: Type = np.float32,
    name: Optional[str] = "VectorFeature",
    preprocess_fn: Optional[Callable[..., Any]] = None
  ) -> None:
    """Init."""
    super().__init__((length,), dtype, name, preprocess_fn)


class OneHotFeature(BaseFeature):
  """One hot feature."""

  def __init__(
    self,
    num_classes: int,
    dtype: Type = np.int8,
    name: Optional[str] = "OneHotFeature",
    preprocess_fn: Optional[Callable[..., Any]] = None
  ) -> None:
    """Init."""
    super().__init__((num_classes,), dtype, name, preprocess_fn)
    self._num_classes = num_classes

  def process(self, inputs: Any) -> Any:
    """Feature process."""
    if self._preprocess_fn is not None:
      inputs = self._preprocess_fn(inputs)
    feature = jnp.array(inputs)
    return jax.nn.one_hot(feature, num_classes=self.num_classes)

  @property
  def num_classes(self) -> int:
    """Num classes."""
    return self._num_classes


class ImageFeature(BaseFeature):
  """Image feature."""

  def __init__(
    self,
    height: int,
    width: int,
    channel: int,
    data_format: str,
    dtype: Type,
    name: Optional[str] = "ImageFeature",
    preprocess_fn: Optional[Callable[..., Any]] = None
  ) -> None:
    """Init."""
    if data_format == "NHWC":
      shape = (height, width, channel)
    elif data_format == "NCHW":
      shape = (channel, height, width)
    else:
      raise ValueError(
        f"data_format value must be `NHWC` or `NCHW`, but got `{data_format}`."
      )
    super().__init__(shape, dtype, name, preprocess_fn)

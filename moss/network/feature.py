"""Feature."""
from typing import Any, Callable, Optional, Tuple, Type

import jax
import jax.numpy as jnp
from dm_env.specs import Array as ArraySpec


class BaseFeature(ArraySpec):
  """Base feature."""

  def __init__(
    self,
    shape: Tuple,
    dtype: Type,
    name: Optional[str] = None,
    process_fn: Callable[..., Any] | None = None
  ) -> None:
    """Init."""
    super().__init__(shape, dtype, name)
    self._process_fn = process_fn

  def process(self, inputs: Any) -> Any:
    """Feature process."""
    if self._process_fn is None:
      return jnp.array(inputs)
    feature = self._process_fn(inputs)
    return jnp.array(feature)


class ScalarFeature(BaseFeature):
  """Scalar feature."""

  def __init__(
    self,
    dtype: Type,
    name: Optional[str] = None,
    process_fn: Callable[..., Any] | None = None
  ) -> None:
    """Init."""
    super().__init__((1,), dtype, name, process_fn)


class VectorFeature(BaseFeature):
  """Vector feature."""

  def __init__(
    self,
    length: int,
    dtype: Type,
    name: Optional[str] = None,
    process_fn: Callable[..., Any] | None = None
  ) -> None:
    """Init."""
    super().__init__((length,), dtype, name, process_fn)


class OneHotFeature(BaseFeature):
  """One hot feature."""

  def __init__(
    self,
    num_classes: int,
    dtype: Type,
    name: str | None = None,
    process_fn: Callable[..., Any] | None = None
  ) -> None:
    """Init."""
    super().__init__((num_classes,), dtype, name, process_fn)
    self._num_classes = num_classes

  def process(self, inputs: Any) -> Any:
    """Feature process."""
    if self._process_fn is None:
      return jnp.array(inputs)
    feature = self._process_fn(inputs)
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
    name: Optional[str] = None,
    process_fn: Callable[..., Any] | None = None
  ) -> None:
    """Init."""
    if data_format == "NHWC":
      shape = (height, width, channel)
    elif data_format == "NCHW":
      shape = (channel, height, width)
    else:
      raise ValueError(
        f"data_format value must be `NHWC` or `NCHW`, but got {data_format}."
      )
    super().__init__(shape, dtype, name, process_fn)

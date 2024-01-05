"""Image feature encoder."""
import collections
from typing import Any, List, Optional

import haiku as hk
import jax

from moss.network.base import Module
from moss.network.layers import ResidualBlock
from moss.types import Array

ResnetConfig = collections.namedtuple(
  "ResnetConfig", ["num_channels", "num_blocks"]
)
Conv2DConfig = collections.namedtuple(
  "Conv2DConfig", ["output_channels", "kernel", "stride", "padding"]
)


class ImageFeatureEncoder(Module):
  """Image featrue encoder."""

  def __init__(
    self,
    name: Optional[str] = None,
    data_format: str = "NHWC",
    use_resnet: bool = False,
    resnet_config: Optional[List[ResnetConfig]] = None,
    conv2d_config: Optional[List[Conv2DConfig]] = None,
    use_orthogonal: bool = True,
  ) -> None:
    """Init.

    Args:
      name: Module name.
      data_format: The data format of the input. Either `NHWC` or `NCHW`. By
        default, `NHWC`.
      use_resnet: Whether use resnet to encoder image feature.
      resnet_config: List of tuple contains 2 nums, num_channels of
        Conv2D and num_blocks of resnet blocks.
      conv2d_config: List of tuple contains 4 arguments (output_channels,
        kernel, stride, padding) of every Conv2D layer.
      use_orthogonal: Whether use orthogonal to initialization params weight.
    """
    super().__init__(name=name)
    self._data_format = data_format
    self._use_resnet = use_resnet
    self._resnet_config = resnet_config
    if self._use_resnet and resnet_config is None:
      raise ValueError(
        "argument `resnet_config` must set when use_resnet is `True`."
      )
    self._conv2d_config = conv2d_config
    if not self._use_resnet and conv2d_config is None:
      raise ValueError(
        "argument `conv2d_config` must set when use_resnet is `False`."
      )
    self._use_orthogonal = use_orthogonal

  def __call__(self, inputs: Array) -> Any:
    """Call."""
    w_init = hk.initializers.Orthogonal() if self._use_orthogonal else None
    if self._use_resnet:
      if self._resnet_config is None:
        raise ValueError(
          "argument `resnet_config` must set when use_resnet is `True`."
        )
      encoder_out = inputs
      for i, (num_channels, num_blocks) in enumerate(self._resnet_config):
        conv = hk.Conv2D(
          num_channels,
          kernel_shape=[3, 3],
          stride=[1, 1],
          w_init=w_init,
          padding="SAME",
          data_format=self._data_format
        )
        encoder_out = conv(encoder_out)  # type: ignore
        encoder_out = hk.max_pool(
          encoder_out,
          window_shape=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding="SAME"
        )
        for j in range(num_blocks):
          block = ResidualBlock(
            num_channels, "residual_{}_{}".format(i, j), self._data_format,
            self._use_orthogonal
          )
          encoder_out = block(encoder_out)
      encoder_out = hk.Flatten()(encoder_out)
    else:
      if self._conv2d_config is None:
        raise ValueError(
          "argument `conv2d_config` must set when use_resnet is `False`."
        )
      layers: List[Any] = []
      for num_channels, kernel, stride, padding in self._conv2d_config:
        layers.append(
          hk.Conv2D(
            num_channels,
            kernel,
            stride,
            padding=padding,
            w_init=w_init,
            data_format=self._data_format
          )
        )
        layers.append(jax.nn.relu)
      layers.append(hk.Flatten())
      encoder = hk.Sequential(layers)
      encoder_out = encoder(inputs)
    return encoder_out

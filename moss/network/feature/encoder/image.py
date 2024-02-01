"""Image feature encoder."""
import collections
from typing import Any, List, Optional

import flax.linen as nn
import jax

from moss.network.feature.encoder.base import FeatureEncoder
from moss.network.layers import ResidualBlock
from moss.network.utils import Flatten
from moss.types import Array

ResnetConfig = collections.namedtuple(
  "ResnetConfig", ["num_channels", "num_blocks"]
)
Conv2DConfig = collections.namedtuple(
  "Conv2DConfig", ["num_channels", "kernel", "strides", "padding"]
)


class ImageEncoder(FeatureEncoder):
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
    self._name = name
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

  def __call__(self, inputs: Array) -> Array:
    """Call."""
    assert self._data_format == "NHWC", ("Only support `NHWC` data format.")
    kernel_init = nn.initializers.orthogonal() if self._use_orthogonal else None
    if self._use_resnet:
      if self._resnet_config is None:
        raise ValueError(
          "argument `resnet_config` must set when use_resnet is `True`."
        )
      encoder_out = inputs
      for i, (num_channels, num_blocks) in enumerate(self._resnet_config):
        conv = nn.Conv(
          num_channels,
          kernel_size=[3, 3],
          strides=[1, 1],
          padding="SAME",
          kernel_init=kernel_init,
        )
        encoder_out = conv(encoder_out)
        encoder_out = nn.max_pool(
          encoder_out,
          window_shape=(3, 3),
          strides=(2, 2),
          padding="SAME",
        )
        for j in range(num_blocks):
          block = ResidualBlock(
            name="residual_{}_{}".format(i, j),
            num_channels=num_channels,
            use_orthogonal=self._use_orthogonal,
          )
          encoder_out = block(encoder_out)
      encoder_out = Flatten()(encoder_out)
    else:
      if self._conv2d_config is None:
        raise ValueError(
          "argument `conv2d_config` must set when use_resnet is `False`."
        )
      layers: List[Any] = []
      for num_channels, kernel, strides, padding in self._conv2d_config:
        layers.append(
          nn.Conv(
            num_channels,
            kernel_size=kernel,
            strides=strides,
            padding=padding,
            kernel_init=kernel_init,
          )
        )
        layers.append(jax.nn.relu)
      layers.append(Flatten())
      encoder = nn.Sequential(layers)
      encoder_out = encoder(inputs)
    return encoder_out

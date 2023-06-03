"""Feature encoder."""
from typing import Any, List, Optional

import haiku as hk
import jax

from moss.types import Array


class CommonEncoder(hk.Module):
  """Common encoder."""

  def __init__(
    self,
    name: str,
    hidden_sizes: List[int],
    use_orthogonal: bool = True
  ) -> None:
    """Init."""
    super().__init__(name)
    self._hidden_sizes = hidden_sizes
    self._use_orthogonal = use_orthogonal

  def __call__(self, inputs: Array) -> Any:
    """Call."""
    layers: List[Any] = []
    w_init = hk.initializers.Orthogonal() if self._use_orthogonal else None
    for hidden_size in self._hidden_sizes:
      layers.append(hk.Linear(hidden_size, w_init=w_init))
      layers.append(jax.nn.relu)
    common_net = hk.Sequential(layers)
    encoder_out = common_net(inputs)
    return encoder_out


class ResidualBlock(hk.Module):
  """Residual block."""

  def __init__(
    self,
    num_channels: int,
    name: Optional[str] = None,
    data_format: str = "NHWC",
    use_orthogonal: bool = True
  ) -> None:
    """Init."""
    super().__init__(name=name)
    self._num_channels = num_channels
    self._data_format = data_format
    self._use_orthogonal = use_orthogonal

  def __call__(self, inputs: Array) -> Any:
    """Call."""
    w_init = hk.initializers.Orthogonal() if self._use_orthogonal else None
    main_branch = hk.Sequential(
      [
        jax.nn.relu,
        hk.Conv2D(
          self._num_channels,
          kernel_shape=[3, 3],
          stride=[1, 1],
          w_init=w_init,
          padding="SAME",
          data_format=self._data_format
        ),
        jax.nn.relu,
        hk.Conv2D(
          self._num_channels,
          kernel_shape=[3, 3],
          stride=[1, 1],
          w_init=w_init,
          padding="SAME",
          data_format=self._data_format
        ),
      ]
    )
    return main_branch(inputs) + inputs


class ImageFeatureEncoder(hk.Module):
  """Image featrue encoder."""

  def __init__(
    self,
    name: Optional[str] = None,
    data_format: str = "NHWC",
    use_resnet: bool = False,
    use_orthogonal: bool = True
  ) -> None:
    """Init.

    Args:
      name: Module name.
      data_format: The data format of the input. Either `NHWC` or `NCHW`. By
        default, `NHWC`.
      use_resnet: Whether use resnet to encoder image feature.
      use_orthogonal: Whether use orthogonal to initialization params weight.
    """
    super().__init__(name=name)
    self._data_format = data_format
    self._use_resnet = use_resnet
    self._use_orthogonal = use_orthogonal

  def __call__(self, inputs: Array) -> Any:
    """Call."""
    w_init = hk.initializers.Orthogonal() if self._use_orthogonal else None
    if self._use_resnet:
      encoder_out = inputs
      for i, (num_channels,
              num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
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
      encoder = hk.Sequential(
        [
          hk.Conv2D(
            32,
            kernel_shape=[8, 8],
            stride=[4, 4],
            w_init=w_init,
            padding="VALID",
            data_format=self._data_format
          ),
          jax.nn.relu,
          hk.Conv2D(
            64,
            kernel_shape=[4, 4],
            stride=[2, 2],
            w_init=w_init,
            padding="VALID",
            data_format=self._data_format
          ),
          jax.nn.relu,
          hk.Conv2D(
            64,
            kernel_shape=[3, 3],
            stride=[1, 1],
            w_init=w_init,
            padding="VALID",
            data_format=self._data_format
          ),
          jax.nn.relu,
          hk.Flatten(),
        ]
      )
      encoder_out = encoder(inputs)
    return encoder_out

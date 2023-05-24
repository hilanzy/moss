"""Feature encoder."""
from typing import Any, Optional

import haiku as hk
import jax

from moss.types import Array


class AtariShallowTorso(hk.Module):
  """Shallow torso for Atari, from the DQN paper."""

  def __init__(self, name: Optional[str] = None):
    """Init."""
    super().__init__(name=name)

  def __call__(self, inputs: Array) -> Any:
    """Call."""
    torso_net = hk.Sequential(
      [
        lambda x: x / 255.,
        hk.Conv2D(32, kernel_shape=[8, 8], stride=[4, 4], padding="VALID"),
        jax.nn.relu,
        hk.Conv2D(64, kernel_shape=[4, 4], stride=[2, 2], padding="VALID"),
        jax.nn.relu,
        hk.Conv2D(64, kernel_shape=[3, 3], stride=[1, 1], padding="VALID"),
        jax.nn.relu,
        hk.Flatten(),
        hk.Linear(512),
        jax.nn.relu,
      ]
    )
    return torso_net(inputs)


class ResidualBlock(hk.Module):
  """Residual block."""

  def __init__(self, num_channels: int, name: Optional[str] = None):
    """Init."""
    super().__init__(name=name)
    self._num_channels = num_channels

  def __call__(self, inputs: Array) -> Any:
    """Call."""
    main_branch = hk.Sequential(
      [
        jax.nn.relu,
        hk.Conv2D(
          self._num_channels, kernel_shape=[3, 3], stride=[1, 1], padding="SAME"
        ),
        jax.nn.relu,
        hk.Conv2D(
          self._num_channels, kernel_shape=[3, 3], stride=[1, 1], padding="SAME"
        ),
      ]
    )
    return main_branch(inputs) + inputs


class AtariDeepTorso(hk.Module):
  """Deep torso for Atari, from the IMPALA paper."""

  def __init__(self, name: Optional[str] = None):
    """Init."""
    super().__init__(name=name)

  def __call__(self, inputs: Array) -> Any:
    """Call."""
    torso_out = inputs / 255.
    for i, (num_channels, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
      conv = hk.Conv2D(
        num_channels, kernel_shape=[3, 3], stride=[1, 1], padding="SAME"
      )
      torso_out = conv(torso_out)  # type: ignore
      torso_out = hk.max_pool(
        torso_out,
        window_shape=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
      )
      for j in range(num_blocks):
        block = ResidualBlock(num_channels, name="residual_{}_{}".format(i, j))
        torso_out = block(torso_out)

    torso_out = jax.nn.relu(torso_out)
    torso_out = hk.Flatten()(torso_out)
    torso_out = hk.Linear(256)(torso_out)
    torso_out = jax.nn.relu(torso_out)
    return torso_out


class ImageFeature(hk.Module):
  """Image featrue encoder."""

  def __init__(
    self, name: Optional[str] = None, use_resnet: bool = False
  ) -> None:
    """Init."""
    super().__init__(name=name)
    self._use_resnet = use_resnet

  def __call__(self, inputs: Array) -> Any:
    """Call."""
    encoder_out = inputs / 255.
    if self._use_resnet:
      for i, (num_channels,
              num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
        conv = hk.Conv2D(
          num_channels, kernel_shape=[3, 3], stride=[1, 1], padding="SAME"
        )
        encoder_out = conv(encoder_out)  # type: ignore
        encoder_out = hk.max_pool(
          encoder_out,
          window_shape=[1, 3, 3, 1],
          strides=[1, 2, 2, 1],
          padding="SAME",
        )
        for j in range(num_blocks):
          block = ResidualBlock(num_channels, name="residual_{}_{}".format(i, j))
          encoder_out = block(encoder_out)
    else:
      encoder = hk.Sequential(
        [
          hk.Conv2D(32, kernel_shape=[8, 8], stride=[4, 4], padding="VALID"),
          jax.nn.relu,
          hk.Conv2D(64, kernel_shape=[4, 4], stride=[2, 2], padding="VALID"),
          jax.nn.relu,
          hk.Conv2D(64, kernel_shape=[3, 3], stride=[1, 1], padding="VALID"),
          jax.nn.relu,
          hk.Flatten(),
        ]
      )
      encoder_out = encoder(inputs)
    return encoder_out

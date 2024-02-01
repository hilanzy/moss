"""Atari network."""
from typing import Any

import numpy as np

from moss.network import CommonNet
from moss.network.action import ActionSpec, DiscreteAction
from moss.network.feature import FeatureSet, FeatureSpec, ImageFeature
from moss.network.feature.encoder import Conv2DConfig, ImageEncoder, ResnetConfig
from moss.network.torso import DenseTorso
from moss.network.value import DenseValue

resnet_default_config = [
  ResnetConfig(16, 2),
  ResnetConfig(16, 2),
  ResnetConfig(32, 2),
]
conv2d_default_config = [
  Conv2DConfig(32, [8, 8], [4, 4], "VALID"),
  Conv2DConfig(64, [4, 4], [2, 2], "VALID"),
  Conv2DConfig(64, [3, 3], [1, 1], "VALID"),
]


def network_maker(
  obs_spec: Any,
  action_spec: Any,
  data_format: str = "NHWC",
  use_resnet: bool = False,
  use_orthogonal: bool = True,
) -> Any:
  """Atari network maker."""
  channel, height, width = obs_spec.obs.shape
  num_actions = action_spec.num_values
  atari_frame = FeatureSet(
    name="atari_frame",
    features={
      "frame":
        ImageFeature(
          height, width, channel, data_format, np.int8, "frame",
          lambda x: x / 255.
        )
    },
    encoder=ImageEncoder(
      name="frame_encoder",
      data_format=data_format,
      use_resnet=use_resnet,
      resnet_config=resnet_default_config,
      conv2d_config=conv2d_default_config,
      use_orthogonal=use_orthogonal
    )
  )
  feature_sets = {
    "atari_frame": atari_frame,
  }
  actions = {
    "atari_action":
      DiscreteAction("atari_action", [512], num_actions, use_orthogonal),
  }

  return CommonNet(
    feature_spec=FeatureSpec(feature_sets),
    action_spec=ActionSpec(actions),
    torso=DenseTorso([512], use_orthogonal),
    value=DenseValue([512, 32], use_orthogonal),
  )

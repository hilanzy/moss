"""PettingZoo network."""
from typing import Any

import numpy as np

from moss.network import CommonNet
from moss.network.action import ActionSpec, DiscreteAction
from moss.network.feature import (
  FeatureSet,
  FeatureSpec,
  ImageFeature,
  OneHotFeature,
)
from moss.network.feature.encoder import (
  CommonEncoder,
  Conv2DConfig,
  ImageFeatureEncoder,
  ResnetConfig,
)
from moss.network.torso import DenseTorso
from moss.network.value import DenseValue

resnet_default_config = [
  ResnetConfig(16, 2),
  ResnetConfig(16, 2),
  ResnetConfig(32, 2),
]
conv2d_default_config = [
  Conv2DConfig(32, 8, 4, "VALID"),
  Conv2DConfig(64, 4, 2, "VALID"),
  Conv2DConfig(64, 3, 1, "VALID"),
]


def network_maker(
  obs_spec: Any,
  action_spec: Any,
  data_format: str = "NHWC",
  use_resnet: bool = False,
  use_orthogonal: bool = True,
) -> Any:
  """Atari network maker."""
  height, width, channel = obs_spec["first_0"].shape
  num_actions = action_spec["first_0"].n

  pettingzoo_frame = FeatureSet(
    name="pettingzoo_frame",
    features={
      "frame":
        ImageFeature(
          height, width, channel, data_format, np.int8, "frame",
          lambda x: x / 255.
        )
    },
    encoder_net=ImageFeatureEncoder(
      "frame_encoder",
      data_format,
      use_resnet=use_resnet,
      resnet_config=resnet_default_config,
      conv2d_config=conv2d_default_config,
      use_orthogonal=use_orthogonal
    )
  )
  player = FeatureSet(
    name="player",
    features={
      "id": OneHotFeature(2, name="id"),
    },
    encoder_net=CommonEncoder(
      "player_encoder",
      hidden_sizes=[32],
      use_orthogonal=use_orthogonal,
    )
  )

  feature_sets = {
    "pettingzoo_frame": pettingzoo_frame,
    "player": player,
  }
  actions = {
    "pettingzoo_action":
      DiscreteAction("pettingzoo_action", [512], num_actions, use_orthogonal),
  }

  return CommonNet(
    feature_spec=FeatureSpec(feature_sets),
    action_spec=ActionSpec(actions),
    torso_net=DenseTorso("torso", [512], use_orthogonal),
    value_net=DenseValue("value", [512, 32], use_orthogonal),
  )

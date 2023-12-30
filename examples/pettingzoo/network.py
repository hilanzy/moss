"""PettingZoo network."""
from functools import partial
from typing import Any

import numpy as np

from moss.network import CTDENet
from moss.network.action import ActionSpec, DiscreteAction
from moss.network.feature import FeatureSet, FeatureSpec, ImageFeature
from moss.network.feature.encoder import (
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
  # channel, height, width = obs_spec.obs.shape
  # num_actions = action_spec.num_values
  channel, height, width = 1, 42, 42
  num_actions = 6

  def frame_feature_set(name):
    feature_set = FeatureSet(
      name=name,
      features={
        "frame":
          ImageFeature(
            height, width, channel, data_format, np.int8, "frame",
            lambda x: x / 255.
          )
      },
      encoder_net_maker=lambda: ImageFeatureEncoder(
        "frame_encoder",
        data_format,
        use_resnet=use_resnet,
        resnet_config=resnet_default_config,
        conv2d_config=conv2d_default_config,
        use_orthogonal=use_orthogonal
      )
    )
    return feature_set

  feature_sets = {
    "pettingzoo_frame": frame_feature_set("pettingzoo_frame"),
  }
  global_feature_sets = {
    "first_frame": frame_feature_set("first_frame"),
    "second_frame": frame_feature_set("second_frame"),
  }
  actions = {
    "pettingzoo_action":
      DiscreteAction("pettingzoo_action", [512], num_actions, use_orthogonal),
  }

  torso_net_maker = partial(DenseTorso, "torso", [512], use_orthogonal)
  value_net_maker = partial(DenseValue, "value", [512, 32], use_orthogonal)
  return CTDENet(
    feature_spec=FeatureSpec(feature_sets),
    global_feature_spec=FeatureSpec(global_feature_sets),
    action_spec=ActionSpec(actions),
    torso_net_maker=torso_net_maker,
    value_net_maker=value_net_maker,
  )

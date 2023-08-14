"""Atari network."""
from functools import partial
from typing import Any

import numpy as np

from moss.network.action import ActionSpec, DiscreteAction
from moss.network.base import CommonNet
from moss.network.feature import (
  FeatureSet,
  FeatureSpec,
  ImageFeature,
  ImageFeatureEncoder,
)
from moss.network.torso import DenseTorso
from moss.network.value import DenseValue


def network_maker(
  obs_spec: Any,
  action_spec: Any,
  data_format: str = "NHWC",
  use_orthogonal: bool = True,
) -> Any:
  """Doom network maker."""
  channel, height, width = obs_spec.obs.shape
  num_actions = action_spec.num_values
  doom_frame = FeatureSet(
    name="doom_frame",
    features={
      "frame":
        ImageFeature(
          height, width, channel, data_format, np.int8, "frame",
          lambda x: x / 255.
        )
    },
    encoder_net_maker=lambda: ImageFeatureEncoder(
      "frame_encoder", data_format, use_orthogonal=use_orthogonal
    )
  )
  feature_spec = {
    "doom_frame": doom_frame,
  }
  actions = {
    "doom_action":
      DiscreteAction("doom_action", [512], num_actions, use_orthogonal)
  }

  torso_net_maker = partial(DenseTorso, "torso", [512], use_orthogonal)
  value_net_maker = partial(DenseValue, "value", [512, 32], use_orthogonal)
  return CommonNet(
    feature_spec=FeatureSpec(feature_spec),
    action_spec=ActionSpec(actions),
    torso_net_maker=torso_net_maker,
    value_net_maker=value_net_maker,
  )

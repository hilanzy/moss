"""Atari network."""
from functools import partial
from typing import Any

import numpy as np

from moss.network.action import DiscreteAction
from moss.network.base import CommonNet
from moss.network.feature import ImageFeature
from moss.network.feature_encoder import ImageFeatureEncoder
from moss.network.feature_set import CommonFeatureSet
from moss.network.torso import DenseTorso
from moss.network.value import DenseValue


def network_maker(
  obs_spec: Any, action_spec: Any, use_orthogonal: bool = True
) -> Any:
  """Atari network maker."""
  channel, height, width = obs_spec.obs.shape
  atari_frame = CommonFeatureSet(
    name="atari_frame",
    features={
      "frame":
        ImageFeature(
          height, width, channel, "NHWC", np.int8, lambda x: x / 255.
        )
    },
    encoder_net_maker=lambda: ImageFeatureEncoder("frame_encoder")
  )
  feature_spec = [atari_frame]
  torso_net_maker = partial(DenseTorso, "torso", [512], use_orthogonal)
  policy_net_maker = partial(
    DiscreteAction, "action", action_spec.num_values, use_orthogonal
  )
  value_net_maker = partial(DenseValue, "value", [512, 32], use_orthogonal)
  return CommonNet(
    feature_spec, torso_net_maker, policy_net_maker, value_net_maker
  )

# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Residual block."""
from typing import Optional

import flax.linen as nn
import jax

from moss.types import Array


class ResidualBlock(nn.Module):
  """Flax residual block module."""
  num_channels: int
  name: Optional[str] = None
  use_orthogonal: bool = True

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Call."""
    kernel_init = nn.initializers.orthogonal() if self.use_orthogonal else None
    main_branch = nn.Sequential(
      [
        jax.nn.relu,
        nn.Conv(
          self.num_channels,
          kernel_size=[3, 3],
          strides=[1, 1],
          padding="SAME",
          kernel_init=kernel_init,
        ), jax.nn.relu,
        nn.Conv(
          self.num_channels,
          kernel_size=[3, 3],
          strides=[1, 1],
          padding="SAME",
          kernel_init=kernel_init,
        )
      ]
    )
    return main_branch(inputs) + inputs

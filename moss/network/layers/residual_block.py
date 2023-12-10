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
from typing import Any, Optional

import haiku as hk
import jax

from moss.types import Array


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

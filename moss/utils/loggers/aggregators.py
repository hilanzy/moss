# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
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
"""Utilities for aggregating to other loggers."""

import time
from collections import defaultdict
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from moss.utils.loggers import base


class Dispatcher(base.Logger):
  """Writes data to multiple `Logger` objects."""

  def __init__(
    self,
    to: Sequence[base.Logger],
    serialize_fn: Optional[Callable[[base.LoggingData], str]] = None,
  ):
    """Initialize `Dispatcher` connected to several `Logger` objects."""
    self._to = to
    self._serialize_fn = serialize_fn

  def write(self, values: base.LoggingData) -> None:
    """Writes `values` to the underlying `Logger` objects."""
    if self._serialize_fn:
      values = self._serialize_fn(values)  # type: ignore
    for logger in self._to:
      logger.write(values)

  def close(self) -> None:
    """Close."""
    for logger in self._to:
      logger.close()


class TimeAggregator(base.Logger):
  """Logger which writes to another logger at a given time interval."""

  def __init__(self, to: base.Logger, time_delta: float):
    """Initializes the logger.

    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
      time_delta: How often to write values out in seconds.
        Note that writes within `time_delta` are stashed.
    """
    self._to = to
    self._values: Dict = defaultdict(list)
    self._time = 0
    self._time_delta = time_delta
    if time_delta < 0:
      raise ValueError(f'time_delta must be greater than 0 (got {time_delta}).')

  def write(self, values: base.LoggingData) -> None:
    """Write."""
    for key, val in values.items():
      self._values[key].append(val)
    now = time.time()
    if (now - self._time) > self._time_delta:
      write_values = {}
      for key in values.keys():
        write_values[key] = np.mean(self._values[key])
        self._values.pop(key)
      self._to.write(write_values)
      self._time = now  # type: ignore

  def close(self) -> None:
    """Close."""
    self._to.close()

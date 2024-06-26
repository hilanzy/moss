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
"""Timestamp logger."""

import time

from moss.utils.loggers import base


class TimestampLogger(base.Logger):
  """Logger which populates the timestamp key with the current timestamp."""

  def __init__(self, logger: base.Logger, timestamp_key: str):
    """Init."""
    self._logger = logger
    self._timestamp_key = timestamp_key

  def write(self, values: base.LoggingData) -> None:
    """Writer."""
    values = dict(values)
    values[self._timestamp_key] = time.time()
    self._logger.write(values)

  def close(self) -> None:
    """Close."""
    self._logger.close()

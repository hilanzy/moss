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
"""Utilities for logging to a TensorboardX SummaryWriter."""

import datetime
import time
from typing import Optional

from absl import logging
from tensorboardX import SummaryWriter

from moss.utils.loggers import base


def _format_key(key: str) -> str:
  """Internal function for formatting keys in TensorboardX format."""
  return key.title().replace('_', '').replace(' ', '')


class TBXSummaryLogger(base.Logger):
  """Logs to a TensorboardX SummaryWriter created in a given logdir.

  If multiple TBXSummaryLogger are created with the same logdir, results will be
  categorized by labels.
  """

  def __init__(
    self, logdir: str, label: str = 'Logs', steps_key: Optional[str] = None
  ):
    """Initializes the logger.

    Args:
      logdir: directory to which we should log files.
      label: label string to use when logging. Default to 'Logs'.
      steps_key: key to use for steps. Must be in the values passed to write.
    """
    self._time = time.time()
    self.label = label
    self._iter = 0
    filename_suffix = f".{datetime.datetime.now().strftime('%f')}"
    self.summary = SummaryWriter(logdir, filename_suffix=filename_suffix)
    self._steps_key = steps_key

  def write(self, values: base.LoggingData) -> None:
    """Write."""
    if self._steps_key is not None and self._steps_key not in values:
      logging.warning('steps key %s not found. Skip logging.', self._steps_key)
      return

    step = values[self._steps_key] if self._steps_key is not None else self._iter

    # TODO(b/159065169): Remove this suppression once the bug is resolved.
    # pytype: disable=unsupported-operands
    for key in values.keys() - [self._steps_key]:
      # pytype: enable=unsupported-operands
      self.summary.add_scalar(
        f'{self.label}/{_format_key(key)}', values[key], step
      )
    self._iter += 1

  def close(self) -> None:
    """Close summary."""
    self.summary.close()

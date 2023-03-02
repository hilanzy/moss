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
"""Logger which makes another logger asynchronous."""

from typing import Any, Mapping

from moss.utils import async_utils
from moss.utils.loggers import base


class AsyncLogger(base.Logger):
  """Logger which makes the logging to another logger asyncronous."""

  def __init__(self, to: base.Logger):
    """Initializes the logger.

    Args:
      to: A `Logger` object to which the current object will forward its results
        when `write` is called.
    """
    self._to = to
    self._async_worker = async_utils.AsyncExecutor(self._to.write, queue_size=5)

  def write(self, values: Mapping[str, Any]) -> None:
    """Write."""
    self._async_worker.put(values)

  def close(self) -> None:
    """Closes the logger, closing is synchronous."""
    self._async_worker.close()
    self._to.close()

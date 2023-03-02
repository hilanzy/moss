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
"""Default logger."""

import logging
from typing import Any, Callable, List, Mapping, Optional

from moss.utils.loggers import aggregators
from moss.utils.loggers import asynchronous as async_logger
from moss.utils.loggers import base, csv, filters, tensorboard, terminal


def make_default_logger(
  label: str,
  use_csv: bool = True,
  csv_log_dir: Optional[str] = None,
  use_tb: bool = True,
  tb_log_dir: Optional[str] = None,
  time_delta: float = 1.0,
  asynchronous: bool = False,
  print_fn: Optional[Callable[[str], None]] = None,
  serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
  steps_key: str = 'steps',
) -> base.Logger:
  """Makes a default Moss logger.

  Args:
    label: Name to give to the logger.
    use_csv: Whether use csv logger to persist data.
    csv_log_dir: Directory for persist csv data.
    use_tb: Whether use tensorboard logger to persist data.
    tb_log_dir: Directory for persist tensorboard data.
    time_delta: Time (in seconds) between logging events.
    asynchronous: Whether the write function should block or not.
    print_fn: How to print to terminal (defaults to print).
    serialize_fn: An optional function to apply to the write inputs before
      passing them to the various loggers.
    steps_key: Ignored.

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  del steps_key
  if not print_fn:
    print_fn = logging.info
  terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

  loggers: List[base.Logger] = [terminal_logger]

  if use_csv:
    if csv_log_dir is None:
      csv_log_dir = "./logs/csv"
    loggers.append(csv.CSVLogger(csv_log_dir, label))
  if use_tb:
    if tb_log_dir is None:
      tb_log_dir = "./logs/tb"
    loggers.append(tensorboard.TFSummaryLogger(tb_log_dir, label))

  # Dispatch to all writers and filter Nones and by time.
  logger = aggregators.Dispatcher(loggers, serialize_fn)
  logger = filters.NoneFilter(logger)
  if asynchronous:
    logger = async_logger.AsyncLogger(logger)  # type: ignore
  logger = filters.TimeFilter(logger, time_delta)

  return logger

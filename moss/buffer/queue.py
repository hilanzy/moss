"""Queue replay buffer."""
from queue import Queue
from typing import List

import jax.numpy as jnp
from jax.tree_util import tree_map

from moss.core import Buffer
from moss.types import Trajectory, Transition


class QueueBuffer(Buffer):
  """Queue replay buffer."""

  def __init__(self, max_size: int, mode: str = "FIFO") -> None:
    """Init.

    Args:
      max_size: Max size of queue buffer.
      mode: Mode of sample and add data(FIFO or LIFO).
    """
    self._queue: Queue[Trajectory] = Queue(max_size)
    self._mode = mode

  def add(self, data: Trajectory) -> None:
    """Add trajectory data to replay buffer.

    Args:
      data: Trajectory data with shape [T, ...].
    """
    self._queue.put(data)

  def sample(self, sample_size: int) -> Transition:
    """Sample trajectory data from replay buffer.

    Returns:
      Batched trajecotry data with shape [T, B, ...].
    """
    data: List[Trajectory] = []
    for _ in range(sample_size):
      traj = self._queue.get()
      data.append(traj)
    stacked_data = tree_map(lambda *x: jnp.stack(x, axis=1), *data)
    return stacked_data

"""Queue replay buffer."""
from queue import LifoQueue, Queue

import jax.numpy as jnp
from jax.tree_util import tree_map

from moss.core import Buffer
from moss.types import Trajectory, Transition


class QueueBuffer(Buffer):
  """Queue replay buffer."""

  def __init__(self, maxsize: int, mode: str = "FIFO") -> None:
    """Init.

    Args:
      maxsize: Max size of queue buffer.
      mode: Mode of sample and add data(FIFO or LIFO).
    """
    if mode == "FIFO":
      self._queue: Queue[Trajectory] = Queue(maxsize)
    elif mode == "LIFO":
      self._queue: Queue[Trajectory] = LifoQueue(maxsize)  # type: ignore
    else:
      raise ValueError(f"mode must be `FIFO` or `LIFO` bug got {mode}.")

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
    data = [self._queue.get() for _ in range(sample_size)]
    stacked_data = tree_map(lambda *x: jnp.stack(x, axis=1), *data)
    return stacked_data

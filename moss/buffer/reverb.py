"""Revber buffer."""
import threading
from typing import Any, Optional

import jax.numpy as jnp
import reverb
from jax.tree_util import tree_map
from reverb import reverb_types

from moss.core import Buffer
from moss.types import Transition


class ReverbQueue(Buffer):
  """Reverb queue buffer."""

  def __init__(
    self,
    max_size: int,
    signature: Optional[reverb_types.SpecNest] = None,
  ) -> None:
    """Init."""
    self._table = reverb.Table.queue("queue", max_size, signature=signature)
    self._server = reverb.Server([self._table])
    self._client = reverb.Client(f"localhost:{self._server.port}")
    self._mutex = threading.Lock()

  def add(self, data: Any) -> None:
    """Add trajectory data to replay buffer.

    Args:
      data: Trajectory data with shape [T, ...].
    """
    with self._mutex:
      with self._client.trajectory_writer(1) as writer:
        writer.append(
          dict(
            step_type=data.step_type,
            state=data.state,
            action=data.action,
            rnn_state=data.rnn_state,
            reward=data.reward,
            policy_logits=data.policy_logits,
            behaviour_value=data.behaviour_value,
          )
        )
        writer.create_item(
          table=self._table.name,
          priority=1.0,
          trajectory=[
            writer.history["step_type"][0],
            writer.history["state"][0],
            writer.history["action"][0],
            writer.history["rnn_state"][0],
            writer.history["reward"][0],
            writer.history["policy_logits"][0],
            writer.history["behaviour_value"][0],
          ]
        )

  def sample(self, sample_size: int) -> Any:
    """Sample trajectory data from replay buffer.

    Returns:
      Batched trajecotry data with shape [T, B, ...].
    """
    sample = self._client.sample(
      self._table.name, sample_size, emit_timesteps=False
    )
    datas = []
    for data in sample:
      transition = Transition(
        step_type=data.data[0],
        state=data.data[1],
        action=data.data[2],
        rnn_state=data.data[3],
        reward=data.data[4],
        policy_logits=data.data[5],
        behaviour_value=data.data[6],
      )
      datas.append(transition)
    stacked_data = tree_map(lambda *x: jnp.stack(x, axis=1), *datas)
    return stacked_data

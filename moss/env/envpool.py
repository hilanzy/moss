"""Envpool environment suite moss."""
from typing import Any, Dict, List

import envpool
import jax
import numpy as np

from moss.env.base import BaseEnv
from moss.types import Array, TimeStep

AgentID = Any


class EnvpoolEnv(BaseEnv):
  """Envpool env."""

  def __init__(self, task_id: str, **kwargs: Any) -> None:
    """Init."""
    self._env = envpool.make_dm(task_id, **kwargs)

  def reset(self) -> Dict[AgentID, TimeStep]:
    """Reset."""
    timestep = self._env.reset()
    return self.timestep_process(timestep)

  def step(self, action: Any) -> Dict[AgentID, TimeStep]:
    """Step."""
    action = self.action_process(action)
    timestep = self._env.step(action)
    return self.timestep_process(timestep)

  def timestep_process(self, timestep: TimeStep) -> Dict[AgentID, TimeStep]:
    """Timestep process."""

    def split_batch_timestep(batch: TimeStep) -> List[TimeStep]:
      """Split batch timestep by env id."""
      size = batch.step_type.size
      timesteps = [
        jax.tree_util.tree_map(lambda x: x[i], batch)  # noqa: B023
        for i in range(size)
      ]
      return timesteps

    timesteps = split_batch_timestep(timestep)
    new_timesteps = {
      agent_id: timestep for agent_id, timestep in enumerate(timesteps)
    }
    return new_timesteps

  def action_process(self, action: Dict[AgentID, Any]) -> Array:
    """Action process."""
    return np.stack(list(action.values()))

  def action_spec(self) -> Any:
    """Get action spec."""
    return self._env.action_spec()

  def observation_spec(self) -> Any:
    """Get observation spec."""
    return self._env.observation_spec()

  def close(self) -> None:
    """Close env and release resources."""
    self._env.close()

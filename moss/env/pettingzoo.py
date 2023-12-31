"""PettingZoo environment suite moss."""
from collections import namedtuple
from typing import Any, Dict, List

from pettingzoo.utils.all_modules import all_environments

from moss.env.base import BaseEnv
from moss.env.wrapper import PettingZooToDeepmindWrapper
from moss.types import TimeStep

AgentID = Any
Observation = namedtuple("Observation", ["obs", "info"])

all_envs = {**all_environments}
for env_id, env_module in all_environments.items():
  all_envs[env_id.split("/")[-1]] = env_module


def make_pettingzoo(
  task_id: str,
  height: int = 84,
  width: int = 84,
  **kwargs: Any
) -> PettingZooToDeepmindWrapper:
  """Pettingzoo env maker."""
  task_id = task_id.replace("-", "_").lower()
  if task_id not in all_envs:
    raise ValueError(f"Not support env `{task_id}`, please check you env name.")
  env_module = all_envs[task_id]
  return PettingZooToDeepmindWrapper(
    env_module.parallel_env, height=height, width=width, **kwargs
  )


class PettingZooEnv(BaseEnv):
  """PetingZoo env."""

  def __init__(
    self,
    task_id: str,
    height: int = 84,
    width: int = 84,
    **kwargs: Any
  ) -> None:
    """Init."""
    self._env = make_pettingzoo(task_id, height, width, **kwargs)

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
    new_timesteps = {}
    for agent_id in self.agents:
      new_timesteps[agent_id] = TimeStep(
        step_type=timestep.step_type[agent_id],
        reward=timestep.reward[agent_id],
        discount=timestep.discount[agent_id],
        observation=Observation(
          timestep.observation.obs[agent_id],
          timestep.observation.info[agent_id],
        ),
      )
    return new_timesteps

  def action_process(self, action: Dict[AgentID, Any]) -> Dict[AgentID, Any]:
    """Action process."""
    return action

  @property
  def agents(self) -> List[AgentID]:
    """Get all agent id."""
    return self._env.agents

  def action_spec(self) -> Any:
    """Get agent action spec by agent id."""
    return self._env.action_spec()

  def observation_spec(self) -> Any:
    """Get agent observation spec by agent id."""
    return self._env.observation_spec()

  def close(self) -> None:
    """Close env and release resources."""
    self._env.close()

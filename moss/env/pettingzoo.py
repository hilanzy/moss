"""PettingZoo environment suite moss."""
from typing import Any, Dict, List

from pettingzoo.utils import ParallelEnv
from pettingzoo.utils.all_modules import all_environments

from moss.env.base import BaseEnv
from moss.env.wrapper import PettingZooToDeepmindWrapper
from moss.types import TimeStep

AgentID = Any

all_envs = {**all_environments}
for env_id, env_module in all_environments.items():
  all_envs[env_id.split("/")[-1]] = env_module


def make_pettingzoo(task_id: str, **kwargs: Any) -> ParallelEnv:
  """Pettingzoo env maker."""
  task_id = task_id.replace("-", "_").lower()
  if task_id not in all_envs:
    raise ValueError(f"Not support env `{task_id}`, please check you env name.")
  env_module = all_envs[task_id]
  return PettingZooToDeepmindWrapper(env_module.parallel_env, **kwargs)


class PettingZooEnv(BaseEnv):
  """PetingZoo env."""

  def __init__(self, task_id: str, **kwargs: Any) -> None:
    """Init."""
    self._env = make_pettingzoo(task_id, **kwargs)

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
        observation=timestep.observation[agent_id],
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
    act_spec = {
      agent_id: self._env.action_space(agent_id) for agent_id in self.agents
    }
    return act_spec

  def observation_spec(self) -> Any:
    """Get agent observation spec by agent id."""
    obs_spec = {
      agent_id: self._env.observation_space(agent_id) for agent_id in self.agents
    }
    return obs_spec

  def close(self) -> None:
    """Close env and release resources."""
    self._env.close()

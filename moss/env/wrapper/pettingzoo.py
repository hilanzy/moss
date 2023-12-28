"""Pettingzoo env wrapper."""
from typing import Any, Callable, Dict, Tuple

from dm_env import StepType, TimeStep
from pettingzoo import ParallelEnv
from pettingzoo.utils.wrappers import BaseParallelWrapper


class AutoResetWrapper(BaseParallelWrapper):
  """Pettingzoo auto reset wrapper."""

  def __init__(self, env: ParallelEnv) -> None:
    """Init."""
    super().__init__(env)
    self._need_reset: bool = False

  def step(self, action: Any) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """Env step."""
    if self._need_reset:
      self._need_reset = False
      obs, info = self.env.reset()
      reward, terminated, truncated = {}, {}, {}
      for agent_id in self.env.agents:
        reward[agent_id] = 0
        terminated[agent_id] = False
        truncated[agent_id] = False
        assert (
          "first_step" not in info[agent_id]
        ), "info dict cannot contain key 'first_step'."
        info[agent_id]["first_step"] = True
    else:
      obs, reward, terminated, truncated, info = self.env.step(action)
      if all(terminated.values()) or all(truncated.values()):
        self._need_reset = True
      for agent_id in self.env.agents:
        assert (
          "first_step" not in info[agent_id]
        ), "info dict cannot contain key 'first_step'."
        info[agent_id]["first_step"] = True

    return obs, reward, terminated, truncated, info


class PettingzooToDeepmindWrapper(object):
  """Wrapper `pettingzoo.ParallelEnv` to `dm_env.Environment`."""

  def __init__(self, env_fn: Callable, **kwargs: Any) -> None:
    """Init."""
    self.env = AutoResetWrapper(env_fn(**kwargs))

  def reset(self, **kwargs: Any) -> TimeStep:
    """Env reset."""
    observation, _ = self.env.reset(**kwargs)
    step_type = {agent_id: StepType.FIRST for agent_id in self.env.agents}
    reward = {agent_id: 0. for agent_id in self.env.agents}
    discount = {agent_id: 1. for agent_id in self.env.agents}
    timestep = TimeStep(
      step_type=step_type,
      reward=reward,
      discount=discount,
      observation=observation,
    )
    return timestep

  def step(self, action: Any) -> TimeStep:
    """Env step."""
    observation, reward, terminated, truncated, info = self.env.step(action)
    step_type, discount = {}, {}
    for agent_id in self.env.agents:
      if terminated[agent_id]:
        step_type[agent_id] = StepType.LAST
        discount[agent_id] = 0.
      elif truncated[agent_id]:
        step_type[agent_id] = StepType.LAST
        discount[agent_id] = 1.
      else:
        step_type[agent_id] =\
          StepType.FIRST if info[agent_id].get("first_step") else StepType.MID
        discount[agent_id] = 1.
    timestep = TimeStep(
      step_type=step_type,
      reward=reward,
      discount=discount,
      observation=observation,
    )
    return timestep

  @property
  def action_spec(self) -> Any:
    """Get action spec."""
    return self.env.action_space

  @property
  def observation_spec(self) -> Any:
    """Get observation spec."""
    return self.env.observation_space

  def close(self) -> None:
    """Close env and release resources."""
    self.env.close()

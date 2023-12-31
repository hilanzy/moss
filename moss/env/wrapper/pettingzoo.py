"""PettingZoo env wrapper."""
from collections import namedtuple
from typing import Any, Callable, Dict, List, Tuple

import gymnasium
import numpy as np
import tree
from dm_env import StepType, TimeStep
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils.wrappers import BaseParallelWrapper

from moss.types import Environment

AgentID = Any
Observation = namedtuple("Observation", ["obs", "info"])


class AutoResetWrapper(BaseParallelWrapper):
  """PettingZoo auto reset wrapper."""

  def __init__(self, env: ParallelEnv) -> None:
    """Init."""
    super().__init__(env)
    self._need_reset: bool = False
    self.agents = self.env.unwrapped.agents

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


class ResizeObservation(BaseParallelWrapper):
  """Resize observation wrapper."""

  def __init__(self, env: ParallelEnv, height: int, width: int) -> None:
    """Init."""
    super().__init__(env)
    self.agents = self.env.agents
    self._shape = (height, width)
    obs_type = self.env.unwrapped.obs_type
    if obs_type == "ram":
      observation_space = gymnasium.spaces.Box(
        low=0, high=255, dtype=np.uint8, shape=(128,)
      )
    else:
      num_channels = 3
      if obs_type == "rgb_image":
        num_channels = 3
      elif obs_type == "grayscale_image":
        num_channels = 1
      observation_space = spaces.Box(
        low=0,
        high=255,
        shape=(height, width, num_channels),
        dtype=np.uint8,
      )
    self.env.unwrapped.observation_spaces = {
      agent: observation_space for agent in self.agents
    }

  def reset(self) -> Tuple[Dict, Dict]:
    """Env reset."""
    obs, info = self.env.reset()
    obs = self.resize_observation(obs)
    return obs, info

  def step(self, actions: dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """Env step."""
    obs, reward, terminated, truncated, info = self.env.step(actions)
    obs = self.resize_observation(obs)
    return obs, reward, terminated, truncated, info

  def resize_observation(self, observation: Dict) -> Dict:
    """Resize observation."""
    try:
      import cv2
    except ImportError as e:
      raise ImportError(
        "opencv (cv2) is not installed, run `pip install opencv-python`"
      ) from e
    observation = tree.map_structure(
      lambda x: cv2.resize(x, self._shape, interpolation=cv2.INTER_AREA),
      observation
    )
    return observation


class PettingZooToDeepmindWrapper(Environment):
  """Wrapper `pettingzoo.ParallelEnv` to `dm_env.Environment`."""

  def __init__(
    self,
    env_fn: Callable,
    height: int = 84,
    width: int = 84,
    **kwargs: Any
  ) -> None:
    """Init."""
    assert (height > 0 and width > 0), (
      f"Expected height and width to be a positive integers, "
      f"got: {height} and {width}"
    )
    # NOTE: AutoReset must be the first wrapper.
    self.env = AutoResetWrapper(env_fn(**kwargs))
    self.env = ResizeObservation(self.env, height, width)
    self._action_spec = {
      agent_id: self.env.action_space(agent_id) for agent_id in self.agents
    }
    self._observation_spec = {
      agent_id: self.env.observation_space(agent_id) for agent_id in self.agents
    }

  def reset(self, **kwargs: Any) -> TimeStep:
    """Env reset."""
    obs, info = self.env.reset(**kwargs)
    for agent_id in self.agents:
      info[agent_id]["agent_id"] = agent_id
    step_type = {agent_id: StepType.FIRST for agent_id in self.env.agents}
    reward = {agent_id: 0. for agent_id in self.env.agents}
    discount = {agent_id: 1. for agent_id in self.env.agents}
    timestep = TimeStep(
      step_type=step_type,
      reward=reward,
      discount=discount,
      observation=Observation(obs, info),
    )
    return timestep

  def step(self, action: Any) -> TimeStep:
    """Env step."""
    obs, reward, terminated, truncated, info = self.env.step(action)
    for agent_id in self.agents:
      info[agent_id]["agent_id"] = agent_id
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
      observation=Observation(obs, info),
    )
    return timestep

  def action_spec(self) -> Any:
    """Get action spec."""
    return self._action_spec

  def observation_spec(self) -> Any:
    """Get observation spec."""
    return self._observation_spec

  @property
  def agents(self) -> List[str]:
    """Get all agent id."""
    return self.env.agents

  def close(self) -> None:
    """Close env and release resources."""
    self.env.close()

"""Gym wrapper."""
from typing import Any, Dict, SupportsFloat, Tuple

import gymnasium as gym
from dm_env import StepType, TimeStep
from gymnasium.core import Env

from moss.types import Environment


class AutoResetWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
  """Auto reset wrapper.

  The different with `gym.AutoResetWrapper` is `gym.AutoResetWrapper` ignore the
  last timestep, and put reward into next episode's first timestep.
  This wrapper follow envpool's auto reset. See more in:
  https://envpool.readthedocs.io/en/latest/content/python_interface.html#auto-reset
  """

  def __init__(self, env: Env) -> None:
    """Init."""
    gym.utils.RecordConstructorArgs.__init__(self)
    gym.Wrapper.__init__(self, env)
    self._need_reset: bool = False

  def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict]:
    """Env step."""
    if self._need_reset:
      self._need_reset = False
      obs, info = self.env.reset()
      reward, terminated, truncated = 0., False, False
      assert (
        "first_step" not in info
      ), "info dict cannot contain key 'first_step'."
      info["first_step"] = True
    else:
      obs, reward, terminated, truncated, info = self.env.step(  # type: ignore
        action
      )
      if terminated or truncated:
        self._need_reset = True
      assert (
        "first_step" not in info
      ), "info dict cannot contain key 'first_step'."
      info["first_step"] = False

    return obs, reward, terminated, truncated, info


class GymToDeepmindWrapper(Environment):
  """Wrapper `gym.Env` to `dm_env.Environment`."""

  def __init__(self, task_id: str, **kwargs: Any) -> None:
    """Init."""
    self._env = AutoResetWrapper(gym.make(task_id, **kwargs))

  def reset(self, **kwargs: Any) -> TimeStep:
    """Env reset."""
    observation, _ = self._env.reset(**kwargs)
    timestep = TimeStep(
      step_type=StepType.FIRST,
      reward=0.,
      discount=1.,
      observation=observation,
    )
    return timestep

  def step(self, action: Any) -> TimeStep:
    """Env step."""
    observation, reward, terminated, truncated, info = self._env.step(action)
    if terminated:
      step_type = StepType.LAST
      discount = 0.0
    elif truncated:
      step_type = StepType.LAST
      discount = 1.0
    else:
      step_type = StepType.FIRST if info.get("first_step") else StepType.MID
      discount = 1.0
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
    return self._env.action_space

  @property
  def observation_spec(self) -> Any:
    """Get observation spec."""
    return self._env.observation_space

  def close(self) -> None:
    """Close env and release resources."""
    self._env.close()

"""Vectorized environments.

Design ideas of vectorized environments refer to the implementation of tianshou.
More about it to see: https://github.com/thu-ml/tianshou
"""
from functools import partial
from typing import Any, Callable, Dict

from moss.env.base import BaseEnv
from moss.env.envpool import EnvpoolEnv
from moss.env.vector.worker import (
  BaseEnvWorker,
  DummyEnvWorker,
  RayEnvWorker,
  SubprocessEnvWorker,
)
from moss.types import TimeStep

EnvID = Any
AgentID = Any


class BaseVectorEnv(object):
  """Base vectorized environments.

  Provide a universal interface to support both vectorized and multi-agent
  environment by use a generic env return format in
  `Dict[EnvID, Dict[AgentID, TimeStep]]`.
  """

  def __init__(
    self,
    num_envs: int,
    env_maker: Callable[[], BaseEnv],
    worker_fn: Callable[[Callable[[], BaseEnv]], BaseEnvWorker],
    **kwargs: Any,
  ) -> None:
    """Init.

    Args:
      num_envs: Num of vectorized environments.
      env_maker: Environment maker function.
      worker_fn: Environment worker function.
      kwargs: Any others env arguments.
    """
    self._num_envs = num_envs
    self._env_maker = env_maker
    self._workers = [worker_fn(env_maker, **kwargs) for _ in range(num_envs)]

  def reset(self) -> Dict[EnvID, Dict[AgentID, TimeStep]]:
    """Resets all environments and returns the initial timestep.

    This method is particularly useful when working with vectorized
    environments,where multiple instances of an environment are being managed
    simultaneously.It ensures that each environment is started from the
    beginning, typically after the end of an episode or during the start of a
    new experiment.

    Returns:
      A dictionary mapping each environment ID (EnvID) to another dictionary,
      which in turn maps each agent ID (AgentID) within that environment to the
      initial time step (TimeStep).The TimeStep object typically includes the
      initial observation, the initial reward (often zero in the first step),
      and other relevant time step information such as whether the state is
      terminal (usually False at the start).

    Example:
      >>> envs = BaseVectorEnv()
      >>> timestpes_dict = envs.reset()
      >>> for env_id, timesteps in timestpes_dict.items():
      ...   for agent_id, timestep in timesteps.items():
      ...     # process the initial timestep for each agent in each environment
      ...     pass
    """
    timesteps_dict = {
      env_id: worker.reset() for env_id, worker in enumerate(self._workers)
    }
    return timesteps_dict

  def step(
    self, action_dict: Dict[EnvID, Dict[AgentID, Any]]
  ) -> Dict[EnvID, Dict[AgentID, TimeStep]]:
    """Vectorized environments step.

    Args:
      action_dict: A dictionary mapping each environment ID (EnvID) to action
      which is a NumPy array, or a nested dict, list or tuple of arrays
      corresponding to `action_spec()`.

    Returns:
      A dictionary mapping each environment ID (EnvID) to another dictionary,
      which in turn maps each agent ID (AgentID) within that environment to the
      next timestep (TimeStep).The TimeStep object typically includes the
      next observation, the next reward (often zero in the first step),
      and other relevant time step information such as whether the state is
      terminal (usually False at the start).
    """
    timesteps_dict = {
      env_id: worker.step(action_dict[env_id])
      for env_id, worker in enumerate(self._workers)
    }
    return timesteps_dict

  def observation_spec(self) -> Any:
    """Defines the observations provided by the environment.

    May use a subclass of `specs.Array` that specifies additional properties
    such as min and max bounds on the values.

    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    pass

  def action_spec(self) -> Any:
    """Defines the actions that should be provided to `step`.

    May use a subclass of `specs.Array` that specifies additional properties
    such as min and max bounds on the values.

    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    pass

  def close(self) -> None:
    """Close all envs and release resources."""
    for worker in self._workers:
      worker.close()

  @property
  def num_envs(self) -> int:
    """Num of vectorized environments."""
    return self._num_envs


class DummyVectorEnv(BaseVectorEnv):
  """Dummy vectorized environments wrapper, implemented in for-loop."""

  def __init__(
    self, num_envs: int, env_maker: Callable[[], BaseEnv], **kwargs: Any
  ) -> None:
    """Dummy vectorized environments wrapper."""
    super().__init__(num_envs, env_maker, DummyEnvWorker, **kwargs)


class EnvpoolVectorEnv(BaseVectorEnv):
  """Envpool vectorized environments warrper, implemented via `envpool`."""

  def __init__(self, task_id: str, **kwargs: Any) -> None:
    """Envpool vectorized environments warrper."""
    env_maker = partial(EnvpoolEnv, task_id, **kwargs)
    super().__init__(1, env_maker, DummyEnvWorker)

  @property
  def num_envs(self) -> int:
    """Num of vectorized environments."""
    return self._workers[0].env.config["num_envs"]


class RayVectorEnv(BaseVectorEnv):
  """Ray vectorized environments wrapper, implemented in ray."""

  def __init__(
    self, num_envs: int, env_maker: Callable[[], BaseEnv], **kwargs: Any
  ) -> None:
    """Ray vectorized environments wrapper."""
    super().__init__(num_envs, env_maker, RayEnvWorker, **kwargs)


class SubprocessVectorEnv(BaseVectorEnv):
  """Subprocess vectorized environment wrapper, implemented in subprocess."""

  def __init__(
    self, num_envs: int, env_maker: Callable[[], BaseEnv], **kwargs: Any
  ) -> None:
    """Subprocess vectorized environment wrapper."""
    super().__init__(num_envs, env_maker, SubprocessEnvWorker, **kwargs)

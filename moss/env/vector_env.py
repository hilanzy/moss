"""Vectorized environments."""
from typing import Any, Callable, Dict, List

from moss.env.base import BaseEnv, TimeStep
from moss.env.worker import BaseEnvWorker, DummyWorker


class BaseVectorEnv(BaseEnv):
  """Base vectorized environments.

  Provide a universal interface to support both vectorized and multi-agent
  environment by repackage all the timesteps into `List[TimeStep]` format.

  For example, for a vectorized and multi-agent environment, we will split
  the multi-agent timesteps for each environment by player id, then pack
  these timesteps into a `TimeStep` list.
  """

  def __init__(
    self,
    num_envs: int,
    env_maker: Callable[[], BaseEnv],
    worker_fn: Callable[[BaseEnv], BaseEnvWorker],
    process_fn: Callable[..., List[TimeStep]],
    **kwargs: Any,
  ) -> None:
    """Init.

    Args:
      num_envs: Num of vectorized environments.
      env_maker: Environment maker function.
      worker_fn: Environment worker function.
      process_fn: Function to package the return timestep of all environments
        into `List[TimeStep]` format.
      kwargs: Any other arguments.
    """
    self._num_envs = num_envs
    self._envs = [env_maker() for _ in range(num_envs)]
    self._workers = [worker_fn(env) for env in self._envs]
    self._process_fn = process_fn
    if kwargs.get("observation_spec") is not None:
      self.observation_spec = kwargs["observation_spec"]  # type: ignore
    if kwargs.get("action_spec") is not None:
      self.action_spec = kwargs["action_spec"]  # type: ignore

  def reset(self) -> Dict[int, List[TimeStep]]:
    """Vectorized environments reset.

    Returns:
      A `TimeStep` list containing all timesteps(split by env_id and player_id)
        of this vectorized multi-agent(maybe) environments.
    """
    timesteps_dict = {
      env_id: self._process_fn(worker.reset())
      for env_id, worker in enumerate(self._workers)
    }
    return timesteps_dict

  def step(self, actions: Any) -> Dict[int, List[TimeStep]]:
    """Vectorized environments step.

    Args:
      actions: A NumPy array, or a nested dict, list or tuple of arrays
      corresponding to `action_spec()`.

    Returns:
      A `TimeStep` list containing:
        All timesteps(split by env_id and player_id) of this vectorized
          multi-agent(maybe) environments.
    """
    timesteps_dict = {
      env_id: self._process_fn(worker.step(actions[env_id]))
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

  @property
  def num_envs(self) -> int:
    """Num of vectorized environments."""
    return self._num_envs


class DummyVectorEnv(BaseVectorEnv):
  """Dummy vectorized environments wrapper, implemented in for-loop."""

  def __init__(
    self, num_envs: int, env_maker: Callable[[], BaseEnv],
    process_fn: Callable[..., List[TimeStep]], **kwargs: Any
  ) -> None:
    """Dummy vectorized environments wrapper."""
    super().__init__(num_envs, env_maker, DummyWorker, process_fn, **kwargs)


class EnvpoolVectorEnv(BaseVectorEnv):
  """Envpool vectorized environments warrper, implemented via `envpool`."""

  def __init__(
    self, env_maker: Callable[[], BaseEnv],
    process_fn: Callable[..., List[TimeStep]], **kwargs: Any
  ) -> None:
    """Envpool vectorized environments warrper."""
    super().__init__(1, env_maker, DummyWorker, process_fn, **kwargs)

  @property
  def num_envs(self) -> int:
    """Num of vectorized environments."""
    return self._envs[0].config["num_envs"]

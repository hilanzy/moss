"""Base environment worker."""
import abc
from typing import Any, Callable

from moss.env.base import BaseEnv

try:
  import ray
except ImportError:
  ray = None  # type: ignore


class BaseEnvWorker(abc.ABC):
  """Base environment worker."""

  def __init__(self, env_maker: Callable[[], BaseEnv]) -> None:
    """Init."""
    self._env = env_maker()

  @abc.abstractmethod
  def reset(self) -> Any:
    """Reset."""

  @abc.abstractmethod
  def step(self, actions: Any) -> Any:
    """Step."""

  @property
  def env(self) -> BaseEnv:
    """Get env."""
    return self._env


class DummyWorker(BaseEnvWorker):
  """Dummy environment worker."""

  def __init__(self, env_maker: Callable[[], BaseEnv]) -> None:
    """Init."""
    super().__init__(env_maker)

  def reset(self) -> Any:
    """Dummy worker reset."""
    return self._env.reset()

  def step(self, actions: Any) -> Any:
    """Dummy worker step."""
    return self._env.step(actions)


class RayEnvWorker(BaseEnvWorker):
  """Ray env worker."""

  def __init__(self, env_maker: Callable[[], BaseEnv]) -> None:
    """Init."""
    if ray is None:
      raise ImportError(
        "Please install ray to support RayVectorEnv: pip install ray"
      )
    self._env =\
      ray.remote(num_cpus=0)(DummyWorker).remote(env_maker)  # type: ignore

  def reset(self) -> Any:
    """Call ray env worker reset remote."""
    return ray.get(self._env.reset.remote())  # type: ignore

  def step(self, actions: Any) -> Any:
    """Call ray env step remote."""
    return ray.get(self._env.step.remote(actions))  # type: ignore

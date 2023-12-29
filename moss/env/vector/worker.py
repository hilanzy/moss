"""Base environment worker."""
import abc
from typing import Any, Callable, Dict

from moss.env.base import BaseEnv
from moss.types import TimeStep

try:
  import ray
except ImportError:
  ray = None  # type: ignore

AgentID = Any


class BaseEnvWorker(abc.ABC):
  """Base environment worker."""

  def __init__(self, env_maker: Callable[[], BaseEnv], **kwargs: Any) -> None:
    """Init."""
    self._env = env_maker(**kwargs)

  @abc.abstractmethod
  def reset(self) -> Dict[AgentID, TimeStep]:
    """Reset."""

  @abc.abstractmethod
  def step(self, actions: Any) -> Dict[AgentID, TimeStep]:
    """Step."""

  @property
  def env(self) -> BaseEnv:
    """Get env."""
    return self._env


class DummyWorker(BaseEnvWorker):
  """Dummy environment worker."""

  def __init__(self, env_maker: Callable[[], BaseEnv], **kwargs: Any) -> None:
    """Init."""
    super().__init__(env_maker, **kwargs)

  def reset(self) -> Any:
    """Dummy worker reset."""
    return self._env.reset()

  def step(self, actions: Any) -> Any:
    """Dummy worker step."""
    return self._env.step(actions)


class RayEnvWorker(BaseEnvWorker):
  """Ray env worker."""

  def __init__(self, env_maker: Callable[[], BaseEnv], **kwargs: Any) -> None:
    """Init."""
    if ray is None:
      raise ImportError(
        "Please install ray to support RayVectorEnv: pip install ray"
      )
    self._env = ray.remote(num_cpus=0)(DummyWorker).remote(
      env_maker, **kwargs
    )  # type: ignore

  def reset(self) -> Any:
    """Call ray env worker reset remote."""
    return ray.get(self._env.reset.remote())  # type: ignore

  def step(self, actions: Any) -> Any:
    """Call ray env step remote."""
    return ray.get(self._env.step.remote(actions))  # type: ignore

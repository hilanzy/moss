"""Base environment worker."""
import abc
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Callable, Dict

from moss.env.base import BaseEnv
from moss.types import TimeStep

try:
  import ray
except ImportError:
  ray = None

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

  @abc.abstractmethod
  def close(self) -> None:
    """Close env and release resources."""

  @property
  def env(self) -> BaseEnv:
    """Get env."""
    return self._env


class DummyEnvWorker(BaseEnvWorker):
  """Dummy environment worker."""

  def __init__(self, env_maker: Callable[[], BaseEnv], **kwargs: Any) -> None:
    """Init."""
    super().__init__(env_maker, **kwargs)

  def reset(self) -> Dict[AgentID, TimeStep]:
    """Dummy worker reset."""
    return self._env.reset()

  def step(self, actions: Any) -> Dict[AgentID, TimeStep]:
    """Dummy worker step."""
    return self._env.step(actions)

  def close(self) -> None:
    """Close env and release resources."""
    self._env.close()


class RayEnvWorker(BaseEnvWorker):
  """Ray env worker."""

  def __init__(self, env_maker: Callable[[], BaseEnv], **kwargs: Any) -> None:
    """Init."""
    if ray is None:
      raise ImportError(
        "Please install ray to support RayVectorEnv: pip install ray"
      )
    self._env =\
      ray.remote(num_cpus=0)(DummyEnvWorker).remote(env_maker, **kwargs)

  def reset(self) -> Dict[AgentID, TimeStep]:
    """Call ray env worker reset remote."""
    return ray.get(self._env.reset.remote())  # type: ignore

  def step(self, actions: Any) -> Dict[AgentID, TimeStep]:
    """Call ray env step remote."""
    return ray.get(self._env.step.remote(actions))  # type: ignore

  def close(self) -> None:
    """Close env and release resources."""
    ray.get(self._env.close.remote())  # type: ignore


class SubprocessEnvWorker(BaseEnvWorker):
  """Subprocess env worker."""

  def __init__(self, env_maker: Callable[[], BaseEnv], **kwargs: Any) -> None:
    """Init."""
    self._parent_remote, self._child_remote = Pipe()
    self._buffer: Any = None
    args = (
      self._parent_remote,
      self._child_remote,
      lambda: env_maker(**kwargs),
    )
    self._process = Process(target=self._worker, args=args, daemon=True)
    self._process.start()
    self._child_remote.close()

  @staticmethod
  def _worker(
    parent: Connection,
    child: Connection,
    env_maker: Callable[[], BaseEnv],
  ) -> None:
    """Process worker."""
    parent.close()
    env = env_maker()
    try:
      while True:
        try:
          cmd, data = child.recv()
        except EOFError:
          child.close()
          break
        if cmd == "reset":
          timesteps = env.reset()
          child.send(timesteps)
        elif cmd == "step":
          timesteps = env.step(data)
          child.send(timesteps)
        elif cmd == "close":
          child.send(env.close())  # type: ignore
          child.close()
        else:
          child.close()
          raise NotImplementedError
    except KeyboardInterrupt:
      child.close()

  def reset(self) -> Dict[AgentID, TimeStep]:
    """Subprocess env worker reset."""
    self._parent_remote.send(["reset", None])
    timesteps = self._parent_remote.recv()
    return timesteps

  def step(self, actions: Any) -> Dict[AgentID, TimeStep]:
    """Subprocess env worker step."""
    self._parent_remote.send(["step", actions])
    timesteps = self._parent_remote.recv()
    return timesteps

  def close(self) -> None:
    """Close env and release resources."""
    try:
      self._parent_remote.send(["close", None])
      self._parent_remote.recv()
      self._process.join()
    except (BrokenPipeError, EOFError, AttributeError):
      pass
    self._process.terminate()

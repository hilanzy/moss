"""Base environment."""
import abc
from typing import Any, Dict

from moss.types import Environment, TimeStep

AgentID = Any


class BaseEnv(Environment):
  """Abstract base environments class for moss."""

  @abc.abstractmethod
  def reset(self) -> Dict[AgentID, TimeStep]:
    """Starts a new environment."""

  @abc.abstractmethod
  def step(self, action: Any) -> Dict[AgentID, TimeStep]:
    """Updates the environment."""

  def send(self, action: Any) -> Any:
    """Send action to low-level environment api."""
    raise NotImplementedError

  def recv(self) -> Any:
    """Receive result from low-level environment api."""
    raise NotImplementedError

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
    """Close env and release resources."""
    pass

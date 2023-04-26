"""Base environment."""
import abc
from typing import Any, NamedTuple

from moss.types import Environment, StepType


class TimeStep(NamedTuple):
  """Returned with every call to `step` and `reset` on an environment.

  `TimeStep` extend from `dm_env.TimeStep` and adds three attributes (env_id,
  player_id, playr_info) to be more general for vectorized and multi-agent
  environment. And we set the default value of the additional attributes as
  `None` to compatible with `dm_env.TimeStep`.

  A `TimeStep` contains the data emitted by an environment at each step
  of interaction. A `TimeStep` holds a `step_type`, an `observation` (typically
  a NumPy array or a dict or list of arrays), and an associated `reward` and
  `discount`.

  The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
  `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
  have `StepType.MID.

  Attributes:
    env_id: A scalar, NumPy array, nested dict, list or tuple of env_id.
    player_id: A scalar, NumPy array, nested dict, list or tuple of player_id.
    player_info: A scalar, NumPy array, nested dict, list or tuple of
      player_info.
    step_type: A `StepType` enum value.
    reward: A scalar, NumPy array, nested dict, list or tuple of rewards; or
      `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
      sequence.
    discount: A scalar, NumPy array, nested dict, list or tuple of discount
      values in the range `[0, 1]`, or `None` if `step_type` is
      `StepType.FIRST`, i.e. at the start of a sequence.
    observation: A NumPy array, or a nested dict, list or tuple of arrays.
      Scalar values that can be cast to NumPy arrays (e.g. Python floats) are
      also valid in place of a scalar array.
  """
  env_id: Any = None
  player_id: Any = None
  player_info: Any = None
  step_type: Any = None
  reward: Any = None
  discount: Any = None
  observation: Any = None

  def first(self) -> bool:
    """Returns whether it is the first timestep."""
    return self.step_type == StepType.FIRST

  def mid(self) -> bool:
    """Returns whether it is the mid timestep."""
    return self.step_type == StepType.MID

  def last(self) -> bool:
    """Returns whether it is the last timestep."""
    return self.step_type == StepType.LAST


class BaseEnv(Environment):
  """Abstract base environments class for moss."""

  @abc.abstractmethod
  def reset(self) -> Any:
    """Starts a new environment."""

  @abc.abstractmethod
  def step(self, action: Any) -> Any:
    """Updates the environment."""

  def send(self, action: Any) -> Any:
    """Send action to low-level environment api."""
    pass

  def recv(self) -> Any:
    """Receive result from low-level environment api."""
    pass

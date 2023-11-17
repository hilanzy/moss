"""Base agent."""
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
from absl import logging

from moss.core import Agent, Buffer, Predictor
from moss.env import TimeStep
from moss.types import AgentState, LoggingData, Reward, StepType, Transition
from moss.utils.loggers import Logger


class BaseAgent(Agent):
  """Base agent."""

  def __init__(
    self,
    name: str,
    unroll_length: int,
    buffer: Buffer,
    predictor: Predictor,
    logger: Logger,
  ) -> None:
    """Init."""
    self._name = name
    self._unroll_length = unroll_length
    self._buffer = buffer
    self._predictor = predictor
    self._logger = logger
    self._unroll_steps = 0
    self._trajectory: List[Transition] = []

  def reset(self) -> LoggingData:
    """Agent reset."""
    raise NotImplementedError

  def step(self, timestep: TimeStep) -> Tuple[AgentState, Reward]:
    """Agent step."""
    raise NotImplementedError

  def inference(self, state: AgentState) -> Any:
    """Agent inference."""
    raise NotImplementedError

  def take_action(self, action: Dict[str, Any]) -> Any:
    """Agent take action."""
    raise NotImplementedError

  def add(self, transition: Transition) -> None:
    """Agent add transition to buffer."""
    is_last_step = transition.step_type == StepType.LAST
    self._trajectory.append(transition)
    self._unroll_steps += 1
    # When unroll_steps == unroll_length + 1 add trajectory to buffer.
    # Because bootstrap need more one step to calculate advantage function.
    if self._unroll_steps > self._unroll_length or is_last_step:
      # Episode end on first trajectory but length less than unroll_length + 1.
      if len(self._trajectory) <= self._unroll_length:
        logging.info(
          "Episode end on first trajectory "
          "but length less than unroll_length + 1."
        )
        self._trajectory = []
        self._unroll_steps = 0
        return

      # Add trajectory to buffer.
      traj = self._trajectory[-(self._unroll_length + 1):]
      stacked_traj = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *traj)
      self._buffer.add(stacked_traj)
      self._trajectory = self._trajectory[-self._unroll_length:]
      self._unroll_steps = 1

      if is_last_step:
        metrics = self.reset()
        self._logger.write(metrics)

"""Doom agent."""
from typing import Any, Dict, Tuple

import jax.numpy as jnp

from moss.agent.base import BaseAgent
from moss.core import Buffer, Predictor
from moss.env import TimeStep
from moss.types import AgentState, LoggingData, Reward
from moss.utils.loggers import Logger


class DoomAgent(BaseAgent):
  """Doom agent."""

  def __init__(
    self, unroll_length: int, buffer: Buffer, predictor: Predictor,
    logger: Logger
  ) -> None:
    """Init."""
    super().__init__("Atari", unroll_length, buffer, predictor, logger)
    self._episode_steps: int = 0
    self._rewards: float = 0

  def _init(self) -> None:
    """Init agent states."""
    self._episode_steps = 0
    self._rewards = 0

  def reset(self) -> LoggingData:
    """Reset agent."""
    metrics = {
      f"{self._name}/episode steps": self._episode_steps,
      f"{self._name}/total rewards": self._rewards
    }
    self._init()
    return metrics

  def step(self, timestep: TimeStep) -> Tuple[AgentState, Reward]:
    """Agent step.

    Return:
      state: agent state input.
        Returns must be serializable Python object to ensure that it can
        exchange data between launchpad's nodes.
    """
    obs = timestep.observation.obs
    obs = jnp.transpose(obs, axes=(1, 2, 0))
    state = {"doom_frame": {"frame": jnp.array(obs)}}
    reward = timestep.reward
    self._episode_steps += 1
    self._rewards += reward
    return state, reward

  def inference(self, state: AgentState) -> Any:
    """Inference."""
    resp_idx = self._predicotr.inference(state)
    return lambda: self._predicotr.result(resp_idx)

  def take_action(self, action: Dict[str, Any]) -> Any:
    """Take action."""
    return action["doom_action"]

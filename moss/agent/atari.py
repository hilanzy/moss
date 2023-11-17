"""Atari agent."""
from typing import Any, Dict, Tuple

import jax.numpy as jnp

from moss.agent.base import BaseAgent
from moss.core import Buffer, Predictor
from moss.env import TimeStep
from moss.types import AgentState, LoggingData, Reward
from moss.utils.loggers import Logger


class AtariAgent(BaseAgent):
  """Atari agent."""

  def __init__(
    self,
    unroll_length: int,
    buffer: Buffer,
    predictor: Predictor,
    logger: Logger,
    data_format: str = "NHWC"
  ) -> None:
    """Init.

    Args:
      unroll_length: Unroll length.
      buffer: Agent replay buffer.
      predictor: Predictor.
      logger: Logger.
      data_format: Atari image data format, must be `NHWC` or `NCHW`, default is
        `NHWC`.
    """
    super().__init__("Atari", unroll_length, buffer, predictor, logger)
    if data_format not in ["NHWC", "NCHW"]:
      raise ValueError(
        f"data_format must be `NHWC` or `NCHW`, but got `{data_format}`."
      )
    self._data_format = data_format
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
    if self._data_format == "NHWC":
      obs = jnp.transpose(obs, axes=(1, 2, 0))
    state = {"atari_frame": {"frame": jnp.array(obs)}}
    reward = timestep.reward
    self._episode_steps += 1
    self._rewards += reward
    return state, reward

  def inference(self, state: AgentState) -> Any:
    """Inference."""
    resp_idx = self._predictor.inference(state)
    return lambda: self._predictor.result(resp_idx)

  def take_action(self, action: Dict[str, Any]) -> Any:
    """Take action."""
    return action["atari_action"]

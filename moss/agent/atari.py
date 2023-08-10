"""Atari agent."""
from typing import Any, Dict, Tuple

import jax.numpy as jnp

from moss.core import Agent, Predictor
from moss.env import TimeStep
from moss.types import AgentState, LoggingData, Reward


class AtariAgent(Agent):
  """Atari agent."""

  def __init__(self, predictor: Predictor, data_format: str = "NHWC") -> None:
    """Init.

    Args:
      predictor: Predictor.
      data_format: Atari image data format, must be `NHWC` or `NCHW`, default is
        `NHWC`.
    """
    self._predicotr = predictor
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
      "agent/episode steps": self._episode_steps,
      "agent/total rewards": self._rewards
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
    resp_idx = self._predicotr.inference(state)
    return lambda: self._predicotr.result(resp_idx)

  def take_action(self, action: Dict[str, Any]) -> Any:
    """Take action."""
    return action["atari_action"]

"""Doom agent."""
from typing import Any, Dict, Tuple

import jax.numpy as jnp

from moss.core import Agent, Predictor
from moss.env import TimeStep
from moss.types import AgentState, LoggingData, Reward


class DoomAgent(Agent):
  """Doom agent."""

  def __init__(self, predictor: Predictor) -> None:
    """Init."""
    self._predicotr = predictor
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

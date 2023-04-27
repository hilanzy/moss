"""Atari agent."""
from typing import Any, Tuple

import jax.numpy as jnp

from moss.core import Agent, Predictor
from moss.env import TimeStep
from moss.types import AgentState, LoggingData, Reward


class AtariAgent(Agent):
  """Atari agent."""

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
    state = jnp.array(obs)
    reward = timestep.reward
    self._episode_steps += 1
    self._rewards += reward
    return state, reward

  def take_action(self, state: AgentState) -> Any:
    """Take action."""
    resp_idx = self._predicotr.inference(state)
    # response = lambda: self._predicotr.result(resp_idx)
    return lambda: self._predicotr.result(resp_idx)

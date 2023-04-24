"""Base agent."""
from typing import Any

import jax.numpy as jnp

from moss.core import Agent, TimeStep
from moss.types import Array, LoggingData


class BaseAgent(Agent):
  """Base agent."""

  def __init__(self) -> None:
    """Init."""
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

  def step(self, timestep: TimeStep) -> Array:
    """Agent step.

    Return:
      obs: agent observation input.
        Returns must be serializable Python object to ensure that it can
        exchange data between launchpad's nodes.
    """
    self._episode_steps += 1
    self._rewards += timestep.reward
    obs = timestep.observation.obs
    obs = jnp.array(obs)
    return obs

  def reward(self, timestep: TimeStep) -> Any:
    """Reward function."""
    return timestep.reward

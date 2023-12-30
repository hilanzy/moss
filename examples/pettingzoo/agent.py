"""PettingZoo agent."""
from typing import Any, Dict, Tuple

import cv2
import jax.numpy as jnp

from moss.agent.base import BaseAgent
from moss.core import Buffer, Predictor
from moss.network.keys import AGENT_STATE, GLOBAL_STATE
from moss.types import LoggingData, Reward, TimeStep
from moss.utils.loggers import Logger


class PettingZooAgent(BaseAgent):
  """PettingZoo agent."""

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
      data_format: PettingZoo image data format, must be `NHWC` or `NCHW`,
        default is `NHWC`.
    """
    super().__init__("PettingZoo", unroll_length, buffer, predictor, logger)
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

  def step(self, timestep: TimeStep) -> Tuple[Dict, Reward]:
    """Agent step.

    Return:
      input_dict: input dict for network.
        Returns must be serializable Python object to ensure that it can
        exchange data between launchpad's nodes.
    """
    obs = timestep.observation[..., -1:]
    if self._data_format == "NCHW":
      obs = jnp.transpose(obs, axes=(2, 1, 0))
    obs = cv2.resize(obs, (42, 42), interpolation=cv2.INTER_AREA)
    obs = jnp.expand_dims(obs, -1)
    state = {"pettingzoo_frame": {"frame": jnp.array(obs)}}
    global_state = {
      "first_frame": {
        "frame": jnp.array(obs)
      },
      "second_frame": {
        "frame": jnp.array(obs)
      },
    }
    input_dict = {AGENT_STATE: state, GLOBAL_STATE: global_state}
    reward = timestep.reward * 1.0
    self._episode_steps += 1
    self._rewards += reward
    return input_dict, reward

  def take_action(self, action: Dict[str, Any]) -> Any:
    """Take action."""
    return action["pettingzoo_action"]

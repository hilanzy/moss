"""Generic actor."""
import collections
import time
from typing import Callable, Dict, Optional, Tuple

import jax
import numpy as np
from absl import logging

from moss.agent import BaseAgent
from moss.core import Actor
from moss.env import BaseVectorEnv
from moss.types import Transition
from moss.utils.loggers import Logger


class GenericActor(Actor):
  """A generic actor implemention.

  A generic actor for all types of environments:
    - Single agent environment.
    - Multi agent environment.
    - Single environment.
    - Vectorized environment.
  """

  def __init__(
    self,
    agent_maker: Callable[..., BaseAgent],
    env_maker: Callable[[], BaseVectorEnv],
    logger_fn: Callable[..., Logger],
    num_trajs: Optional[int] = None,
  ) -> None:
    """Init."""
    self._agent_maker = agent_maker
    self._env_maker = env_maker
    self._num_trajs = num_trajs
    self._logger_fn = logger_fn
    self._logger = logger_fn(label="Actor")
    logging.info(jax.devices())

  def run(self) -> None:
    """Run actor."""
    num_trajs = 0
    agent_logger = self._logger_fn(label="Agent")
    agents: Dict[Tuple[int, int], BaseAgent] = {}
    envs = self._env_maker()
    timesteps_dict = envs.reset()
    while not self._num_trajs or num_trajs < self._num_trajs:
      actor_step_start = time.time()
      states_dict = collections.defaultdict(list)
      rewards_dict = collections.defaultdict(list)
      responses_dict = collections.defaultdict(list)
      for env_id, timesteps in timesteps_dict.items():
        for timestep in timesteps:
          ep_id = (env_id, timestep.player_id)
          if ep_id not in agents.keys():
            agents[ep_id] = self._agent_maker(timestep.player_info, agent_logger)
          state, reward = agents[ep_id].step(timestep)
          response = agents[ep_id].inference(state)
          states_dict[env_id].append(state)
          rewards_dict[env_id].append(reward)
          responses_dict[env_id].append(response)
      get_result_time = 0.0
      actions_dict = collections.defaultdict(list)
      for env_id, timesteps in timesteps_dict.items():
        for timestep, state, response, reward in zip(
          timesteps, states_dict[env_id], responses_dict[env_id],
          rewards_dict[env_id]
        ):
          ep_id = (env_id, timestep.player_id)
          get_result_start = time.time()
          action, logits, value, rnn_state = agents[ep_id].result(response)
          get_result_time += time.time() - get_result_start
          take_action = agents[ep_id].take_action(action)
          actions_dict[env_id].append(take_action)
          transition = Transition(
            step_type=timestep.step_type,
            state=state,
            action=action,
            rnn_state=rnn_state,
            reward=reward,
            policy_logits=logits,
            behaviour_value=value,
          )
          agents[ep_id].add(transition)
      actions = {
        env_id: np.stack(actions) for env_id, actions in actions_dict.items()
      }
      envs_step_start = time.time()
      timesteps_dict = envs.step(actions)
      self._logger.write(
        {
          "time/get result": get_result_time,
          "time/envs step": time.time() - envs_step_start,
          "time/actor step": time.time() - actor_step_start,
        }
      )

"""Generic actor."""
import collections
import time
from typing import Callable, Dict, Tuple

import jax
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
  ) -> None:
    """Init."""
    self._agent_maker = agent_maker
    self._env_maker = env_maker
    self._logger_fn = logger_fn
    self._logger = logger_fn(label="Actor")
    logging.info(jax.devices())

  def run(self) -> None:
    """Run actor."""
    agent_logger = self._logger_fn(label="Agent")
    agents: Dict[Tuple[int, int], BaseAgent] = {}
    envs = self._env_maker()
    timesteps_dict = envs.reset()
    while True:
      actor_step_start = time.time()
      input_dicts = collections.defaultdict(list)
      rewards = collections.defaultdict(list)
      responses = collections.defaultdict(list)
      for env_id, timesteps in timesteps_dict.items():
        for player_id, timestep in timesteps.items():
          # NOTE: To avoid confusion between environment agent and rl agent,
          # we refer to the agent_id as a player_id.
          ep_id = (env_id, player_id)
          if ep_id not in agents.keys():
            agents[ep_id] = self._agent_maker(timestep, agent_logger)
          input_dict, reward = agents[ep_id].step(timestep)
          response = agents[ep_id].inference(input_dict)
          input_dicts[env_id].append(input_dict)
          rewards[env_id].append(reward)
          responses[env_id].append(response)
      get_result_time = 0.0
      action_dict = collections.defaultdict(dict)  # type: ignore
      for env_id, timesteps in timesteps_dict.items():
        for (player_id, timestep), input_dict, response, reward in zip(
          timesteps.items(), input_dicts[env_id], responses[env_id],
          rewards[env_id]
        ):
          ep_id = (env_id, player_id)
          get_result_start = time.time()
          action, logits, value, rnn_state = agents[ep_id].result(response)
          get_result_time += time.time() - get_result_start
          take_action = agents[ep_id].take_action(action)
          action_dict[env_id][player_id] = take_action
          transition = Transition(
            step_type=timestep.step_type,
            input_dict=input_dict,
            action=action,
            rnn_state=rnn_state,
            reward=reward,
            policy_logits=logits,
            behaviour_value=value,
          )
          agents[ep_id].add(transition)
      envs_step_start = time.time()
      timesteps_dict = envs.step(action_dict)
      self._logger.write(
        {
          "time/get result": get_result_time,
          "time/envs step": time.time() - envs_step_start,
          "time/actor step": time.time() - actor_step_start,
        }
      )

"""A actor for vectorized environment."""
import collections
import time
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging

from moss.core import Actor, Agent, Buffer
from moss.env import BaseVectorEnv
from moss.types import Transition
from moss.utils.loggers import Logger


class VectorActor(Actor):
  """Base actor."""

  def __init__(
    self,
    buffer: Buffer,
    agent_maker: Callable[..., Agent],
    env_maker: Callable[[], BaseVectorEnv],
    unroll_len: int,
    logger_fn: Callable[..., Logger],
    num_trajs: Optional[int] = None,
  ) -> None:
    """Init."""
    self._buffer = buffer
    self._agent_maker = agent_maker
    self._env_maker = env_maker
    self._unroll_len = unroll_len
    self._num_trajs = num_trajs
    self._logger = logger_fn(label="Actor")
    logging.info(jax.devices())

  def run(self) -> None:
    """Run actor."""
    num_trajs = 0
    unroll_len = self._unroll_len + 1
    unroll_steps: Dict[Tuple[int, int], int] = collections.defaultdict(int)
    trajs: Dict[Tuple[int, int],
                List[Transition]] = collections.defaultdict(list)
    agents: Dict[Tuple[int, int], Agent] = {}
    envs = self._env_maker()
    timesteps_dict = envs.reset()
    while not self._num_trajs or num_trajs < self._num_trajs:
      actor_step_start = time.time()
      states_dict = collections.defaultdict(list)
      rewards_dict = collections.defaultdict(list)
      responses_dict = collections.defaultdict(list)
      actions_dict = collections.defaultdict(list)
      for env_id, timesteps in timesteps_dict.items():
        for timestep in timesteps:
          ep_id = (env_id, timestep.player_id)
          if ep_id not in agents.keys():
            agents[ep_id] = self._agent_maker(timestep.player_info)
          state, reward = agents[ep_id].step(timestep)
          response = agents[ep_id].take_action(state)
          states_dict[env_id].append(state)
          rewards_dict[env_id].append(reward)
          responses_dict[env_id].append(response)
      get_result_start = time.time()
      results = collections.defaultdict(list)
      for env_id, responses in responses_dict.items():
        for response in responses:
          results[env_id].append(response())
      get_result_time = time.time() - get_result_start
      for env_id in timesteps_dict.keys():
        for timestep, state, (action, logits), reward in zip(
          timesteps_dict[env_id], states_dict[env_id], results[env_id],
          rewards_dict[env_id]
        ):
          ep_id = (env_id, timestep.player_id)
          actions_dict[env_id].append(action)
          transition = Transition(
            step_type=timestep.step_type,
            state=state,
            action=action,
            policy_logits=logits,
            reward=reward,
          )
          trajs[ep_id].append(transition)
          unroll_steps[ep_id] += 1
          if unroll_steps[ep_id] >= unroll_len or timestep.last():
            if timestep.last():
              metrics = agents[ep_id].reset()
              self._logger.write(metrics)

            # Episode end on first trajectory but length less than unroll_len.
            if len(trajs[ep_id]) < unroll_len:
              logging.info(
                "Episode end on first trajectory "
                "but length less than unroll_len."
              )
              trajs[ep_id] = []
              continue

            traj = trajs[ep_id][-unroll_len:]
            stacked_traj = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *traj)
            self._buffer.add(stacked_traj)
            trajs[ep_id] = trajs[ep_id][-unroll_len:]
            unroll_steps[ep_id] = 1
            num_trajs += 1

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

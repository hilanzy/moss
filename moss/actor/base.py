"""Base actor."""
import collections
import time
from typing import Callable, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging

from moss.core import Actor, Agent, Buffer, Predictor
from moss.types import Environment, TimeStep, Transition
from moss.utils.loggers import Logger


class BaseActor(Actor):
  """Base actor."""

  def __init__(
    self,
    buffer: Buffer,
    agent_maker: Callable[[], Agent],
    env_maker: Callable[[], Environment],
    predictor: Predictor,
    unroll_len: int,
    logger_fn: Callable[..., Logger],
    num_trajs: Optional[int] = None,
  ) -> None:
    """Init."""
    self._buffer = buffer
    self._agent_maker = agent_maker
    self._env_maker = env_maker
    self._predictor = predictor
    self._unroll_len = unroll_len
    self._num_trajs = num_trajs
    self._logger = logger_fn(label="Actor")
    logging.info(jax.devices())

  def _split_batch_timestep(self, batch: TimeStep) -> List[TimeStep]:
    """Split batch timestep by env id."""
    size = batch.step_type.size
    timesteps = [
      jax.tree_util.tree_map(lambda x: x[i], batch)  # noqa: B023
      for i in range(size)
    ]
    return timesteps

  def run(self) -> None:
    """Run actor."""
    envs = self._env_maker()
    batch_timestep = envs.reset()
    num_envs = envs.config["num_envs"]
    agents = [self._agent_maker() for _ in range(num_envs)]
    num_trajs = 0
    unroll_len = self._unroll_len + 1
    unroll_steps: Dict[int, int] = collections.defaultdict(int)
    trajs: Dict[int, List[Transition]] = collections.defaultdict(list)

    while not self._num_trajs or num_trajs < self._num_trajs:
      actor_step_start = time.time()
      observations, responses, rewards = [], [], []
      timesteps = self._split_batch_timestep(batch_timestep)
      for env_id, timestep in enumerate(timesteps):
        observation = agents[env_id].step(timestep)
        response = self._predictor.inference(observation)
        reward = agents[env_id].reward(timestep)
        observations.append(observation)
        responses.append(response)
        rewards.append(reward)

      actions = []
      get_result_start = time.time()
      responses_results = [self._predictor.result(resp) for resp in responses]
      get_result_time = time.time() - get_result_start
      for env_id, (timestep, observation, (action, logits), reward) in enumerate(
        zip(timesteps, observations, responses_results, rewards)
      ):
        actions.append(action)
        transition = Transition(
          step_type=timestep.step_type,
          obs=observation,
          action=action,
          policy_logits=logits,
          reward=reward,
        )
        trajs[env_id].append(transition)
        unroll_steps[env_id] += 1
        if unroll_steps[env_id] >= unroll_len or timestep.last():
          if timestep.last():
            metrics = agents[env_id].reset()
            self._logger.write(metrics)

          # Episode end on first trajectory but length less than unroll_len.
          if len(trajs[env_id]) < unroll_len:
            logging.info(
              "Episode end on first trajectory but length less than unroll_len."
            )
            trajs[env_id] = []
            continue

          traj = trajs[env_id][-unroll_len:]
          stacked_traj = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *traj)
          self._buffer.add(stacked_traj)
          trajs[env_id] = trajs[env_id][-unroll_len:]
          unroll_steps[env_id] = 1
          num_trajs += 1

      actions = np.stack(actions)
      envs_step_start = time.time()
      batch_timestep = envs.step(actions)
      self._logger.write(
        {
          "time/get result": get_result_time,
          "time/envs step": time.time() - envs_step_start,
          "time/actor step": time.time() - actor_step_start,
        }
      )

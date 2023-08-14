"""Atari local run."""
import os
import pickle
from functools import partial

import jax
import numpy as np
import tree
from absl import app, flags, logging
from dm_env import TimeStep

from examples.atari.network import network_maker
from examples.atari.utils import LocalEnv
from moss.agent.atari import AtariAgent
from moss.predictor.base import BasePredictor
from moss.types import Params
from moss.utils.loggers import TerminalLogger

flags.DEFINE_string("task_id", "Pong-v5", "Task name.")
flags.DEFINE_string("model_path", None, "Restore model path.")

FLAGS = flags.FLAGS


def main(_):
  """Main."""
  local_env = LocalEnv(FLAGS.task_id)
  obs_spec = local_env.observation_spec()
  action_spec = local_env.action_spec()

  predictor = BasePredictor(
    1, partial(network_maker, obs_spec, action_spec), TerminalLogger
  )
  with open(FLAGS.model_path, mode="rb") as f:
    params: Params = pickle.load(f)
    predictor.update_params(params)
  agent = AtariAgent(predictor)

  rng = jax.random.PRNGKey(42)
  total_reward = 0
  timestep: TimeStep = local_env.reset()
  while True:
    local_env.render()
    if timestep.first():
      total_reward = 0
      agent.reset()

    state, reward = agent.step(timestep)
    total_reward += reward
    sub_key, rng = jax.random.split(rng)
    state = tree.map_structure(lambda x: np.expand_dims(x, 0), state)
    action, _ = predictor._forward(params, state, sub_key)
    take_action = agent.take_action(action)
    take_action = np.array(take_action)
    timestep = local_env.step(take_action)

    if timestep.last():
      _, reward = agent.step(timestep)
      total_reward += reward
      logging.info(f"Total reward: {total_reward}")


if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  app.run(main)

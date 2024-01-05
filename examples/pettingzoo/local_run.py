"""Atari local run."""
import os
import pickle
from functools import partial
from typing import Any, Dict

import jax
import numpy as np
import tree
from absl import app, flags, logging
from dm_env import TimeStep

from examples.pettingzoo.agent import PettingZooAgent
from examples.pettingzoo.network import network_maker
from moss.env.pettingzoo import PettingZooEnv
from moss.predictor import BasePredictor
from moss.types import Params
from moss.utils.loggers import TerminalLogger

flags.DEFINE_string("task_id", "pong_v3", "Task name.")
flags.DEFINE_string("model_path", None, "Restore model path.")
flags.DEFINE_bool(
  "use_resnet", False, "Use resnet or conv2d encoder for atari image."
)

FLAGS = flags.FLAGS


def main(_):
  """Main."""
  local_env = PettingZooEnv(FLAGS.task_id, render_mode="human")
  obs_spec = local_env.observation_spec()
  action_spec = local_env.action_spec()

  use_resnet = FLAGS.use_resnet
  predictor = BasePredictor(
    1, partial(network_maker, obs_spec, action_spec, use_resnet=use_resnet),
    TerminalLogger
  )
  with open(FLAGS.model_path, mode="rb") as f:
    params: Params = pickle.load(f)
    predictor.update_params(params)
  agent = PettingZooAgent(0, None, predictor, TerminalLogger())

  rng = jax.random.PRNGKey(42)
  timestep_dict: Dict[Any, TimeStep] = local_env.reset()
  rewards, actions = {}, {}
  while True:
    for agent_id, timestep in timestep_dict.items():
      if timestep.first():
        rewards[agent_id] = 0
        agent.reset()

      input_dict, reward = agent.step(timestep)
      rewards[agent_id] += reward
      sub_key, rng = jax.random.split(rng)
      input_dict = tree.map_structure(lambda x: np.expand_dims(x, 0), input_dict)
      action, _ = predictor._forward(params, input_dict, None, sub_key)
      take_action = agent.take_action(action)
      actions[agent_id] = int(take_action[0])

      if timestep.last():
        _, reward = agent.step(timestep)
        rewards[agent_id] += reward
        logging.info(f"{agent_id} total reward: {rewards[agent_id]}")
    timestep_dict = local_env.step(actions)


if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  app.run(main)

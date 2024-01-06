"""Doom local run."""
import os
import pickle
from functools import partial

import jax
import numpy as np
import tree
from absl import app, flags, logging

from examples.vizdoom.agent import DoomAgent
from examples.vizdoom.network import network_maker
from examples.vizdoom.utils import LocalEnv
from moss.predictor import BasePredictor
from moss.types import Params, TimeStep
from moss.utils.loggers import TerminalLogger

flags.DEFINE_string("task_id", "D1Basic-v1", "Task name.")
flags.DEFINE_string(
  "cfg_path", "examples/vizdoom/maps/D1_basic.cfg", "Task config path."
)
flags.DEFINE_string(
  "wad_path", "examples/vizdoom/maps/D1_basic.wad", "Map config path."
)
flags.DEFINE_string("model_path", None, "Restore model path.")

FLAGS = flags.FLAGS


def main(_):
  """Main."""
  reward_config = {
    "KILLCOUNT": [20.0, -20.0],
    "HEALTH": [1.0, 0.0],
    "AMMO2": [1.0, -1.0],
  }
  if "battle" in FLAGS.task_id:
    reward_config["HEALTH"] = [1.0, -1.0]
  local_env = LocalEnv(
    FLAGS.task_id,
    scale=2,
    cfg_path=FLAGS.cfg_path,
    wad_path=FLAGS.wad_path,
    reward_config=reward_config,
    use_combined_action=True,
    max_episode_steps=2625,
    use_inter_area_resize=False,
  )
  obs_spec = local_env.observation_spec()
  action_spec = local_env.action_spec()

  predictor = BasePredictor(
    1, partial(network_maker, obs_spec, action_spec), TerminalLogger
  )
  with open(FLAGS.model_path, mode="rb") as f:
    params: Params = pickle.load(f)
    predictor.update_params(params)
  agent = DoomAgent(0, None, predictor, TerminalLogger())

  rng = jax.random.PRNGKey(42)
  total_reward = 0
  timestep: TimeStep = local_env.reset()
  while True:
    if timestep.first():
      total_reward = 0
      agent.reset()

    input_dict, reward = agent.step(timestep)
    total_reward += reward
    sub_key, rng = jax.random.split(rng)
    input_dict = tree.map_structure(lambda x: np.expand_dims(x, 0), input_dict)
    action, _ = predictor._forward(params, input_dict, None, sub_key)
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

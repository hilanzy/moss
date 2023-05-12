"""Atari local run."""
import pickle

import envpool
import jax
from absl import app, flags, logging
from dm_env import TimeStep

from moss.agent.atari import AtariAgent
from moss.network.base import SimpleNet
from moss.predictor.base import BasePredictor
from moss.types import Params
from moss.utils.loggers import make_default_logger

flags.DEFINE_string("task_id", "Pong-v5", "Task name.")
flags.DEFINE_string("model_path", None, "Restore model path.")

FLAGS = flags.FLAGS


def main(_):
  """Main."""
  local_env = envpool.make_dm(FLAGS.task_id)
  obs_sepc = local_env.observation_spec()
  action_sepc = local_env.action_spec()

  def network_maker() -> SimpleNet:
    """Network maker."""
    return SimpleNet(obs_sepc, action_sepc)

  def logger_fn():
    """Logger function."""
    return make_default_logger("local run", False, False)

  predictor = BasePredictor(1, network_maker, logger_fn)
  with open(FLAGS.model_path, mode="rb") as f:
    params: Params = pickle.load(f)
    predictor.update_params(params)
  agent = AtariAgent(predictor)

  rng = jax.random.PRNGKey(42)
  total_reward = 0
  timestep: TimeStep = local_env.reset()
  while True:
    if timestep.first():
      total_reward = 0
      agent.reset()

    state, reward = agent.step(timestep)
    total_reward += reward
    sub_key, rng = jax.random.split(rng)
    action, _ = predictor._forward(params, state, sub_key)
    timestep = local_env.step(action)

    if timestep.last():
      logging.info(f"Total reward: {total_reward}")


if __name__ == "__main__":
  app.run(main)

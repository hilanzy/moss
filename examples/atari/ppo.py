"""Atati ppo example."""
import os
from functools import partial
from typing import Any, Callable, List

import envpool
import jax
import launchpad as lp
from absl import app, flags, logging
from launchpad.nodes.dereference import Deferred
from launchpad.nodes.python.local_multi_processing import PythonProcess

from examples.atari.network import network_maker
from moss.actor.vector import VectorActor
from moss.agent.atari import AtariAgent
from moss.buffer.queue import QueueBuffer
from moss.env import EnvpoolVectorEnv, TimeStep
from moss.learner.ppo import PPOLearner
from moss.predictor.base import BasePredictor
from moss.types import Environment
from moss.utils.loggers import Logger, experiment_logger_factory
from moss.utils.paths import get_unique_id

flags.DEFINE_string("task_id", "Pong-v5", "Task name.")
flags.DEFINE_integer("stack_num", 1, "Stack nums.")
flags.DEFINE_integer("num_envs", 32, "Num of envs.")
flags.DEFINE_integer("num_threads", 6, "Num threads of envs.")
flags.DEFINE_bool(
  "use_orthogonal", True, "Use orthogonal to initialization params weight."
)
flags.DEFINE_integer("num_actors", 16, "Num of actors.")
flags.DEFINE_integer("num_predictors", 4, "Num of predictors.")
flags.DEFINE_integer("unroll_len", 16, "Unroll length.")
flags.DEFINE_integer("predict_batch_size", 32, "Predict batch size.")
flags.DEFINE_integer("training_batch_size", 64, "Training batch size.")
flags.DEFINE_string("model_path", None, "Restore model path.")
flags.DEFINE_integer(
  "publish_interval", 1, "Publish params to predicotrs interval(by train steps)."
)
flags.DEFINE_integer("save_interval", 500, "Save interval(by train steps).")
flags.DEFINE_float("learning_rate", 5e-4, "Learning rate.")
flags.DEFINE_float("gamma", 0.99, "Reward discount rate.")
flags.DEFINE_float("gae_lambda", 0.95, "GAE lambda.")
flags.DEFINE_float(
  "pg_clip_epsilon", 0.1, "Epsilon for policy gradient clipping."
)
flags.DEFINE_float("value_clip_epsilon", 0.2, "Epsilon for value clipping.")
flags.DEFINE_float("critic_coef", 0.25, "Critic coefficient.")
flags.DEFINE_float("entropy_coef", 0.01, "Entropy coefficient.")
flags.DEFINE_integer("buffer_size", 2048, "Replay buffer size.")
flags.DEFINE_string(
  "buffer_mode", "FIFO", "Queue buffer mode(Support `FIFO` and `LIFO`)."
)
flags.DEFINE_string("lp_launch_type", "local_mp", "Launch type.")

FLAGS = flags.FLAGS


def make_lp_program() -> Any:
  """Make launchpad program."""
  (exp_uid,) = get_unique_id()
  task_id = FLAGS.task_id
  stack_num = FLAGS.stack_num
  num_envs = FLAGS.num_envs
  num_threads = FLAGS.num_threads
  unroll_len = FLAGS.unroll_len

  dummy_env: Environment = envpool.make_dm(
    task_id, stack_num=stack_num, num_envs=1
  )
  obs_spec: Any = dummy_env.observation_spec()
  action_spec: Any = dummy_env.action_spec()
  use_orthogonal = FLAGS.use_orthogonal

  logging.info(f"Task id: {task_id}")
  logging.info(f"Observation shape: {obs_spec.obs.shape}")
  logging.info(f"Action space: {action_spec.num_values}")

  def env_maker() -> EnvpoolVectorEnv:
    """Env maker."""

    def env_wrapper() -> Any:
      """Env wrapper."""
      return envpool.make_dm(
        task_id, stack_num=stack_num, num_envs=num_envs, num_threads=num_threads
      )

    def process_fn(timesteps: Any) -> Any:
      """Timesteps process function."""

      def split_batch_timestep(batch: TimeStep) -> List[TimeStep]:
        """Split batch timestep by env id."""
        size = batch.step_type.size
        timesteps = [
          jax.tree_util.tree_map(lambda x: x[i], batch)  # noqa: B023
          for i in range(size)
        ]
        return timesteps

      timesteps = split_batch_timestep(timesteps)
      new_timesteps = []
      for i, timestep in enumerate(timesteps):
        new_timesteps.append(TimeStep(0, i, None, *timestep))
      return new_timesteps

    return EnvpoolVectorEnv(env_wrapper, process_fn)

  def agent_maker(buffer: QueueBuffer, predictor: BasePredictor) -> Callable:
    """Agent maker."""

    def agent_wrapper(player_info: Any, logger: Logger) -> AtariAgent:
      """Return a agent."""
      del player_info
      return AtariAgent(unroll_len, buffer, predictor, logger)

    return agent_wrapper

  logger_fn = experiment_logger_factory(
    project=task_id, uid=exp_uid, time_delta=2.0, print_fn=logging.info
  )

  program = lp.Program("ppo")
  with program.group("buffer"):
    buffer_node = lp.CourierNode(
      QueueBuffer,
      FLAGS.buffer_size,
      FLAGS.buffer_mode,
    )
    buffer = program.add_node(buffer_node)

  predictors = []
  with program.group("predictor"):
    for _ in range(FLAGS.num_predictors):
      predictor_node = lp.CourierNode(
        BasePredictor,
        FLAGS.predict_batch_size,
        partial(network_maker, obs_spec, action_spec, "NHWC", use_orthogonal),
        logger_fn,
      )
      predictor = program.add_node(predictor_node)
      predictors.append(predictor)

  with program.group("actor"):
    for i in range(FLAGS.num_actors):
      actor_node = lp.CourierNode(
        VectorActor,
        Deferred(agent_maker, buffer, predictors[i % FLAGS.num_predictors]),
        env_maker,
        logger_fn,
      )
      program.add_node(actor_node)

  with program.group("learner"):
    save_path = os.path.join("checkpoints", task_id, exp_uid)
    learner_node = lp.CourierNode(
      PPOLearner,
      buffer,
      predictors,
      partial(network_maker, obs_spec, action_spec, "NHWC", use_orthogonal),
      logger_fn,
      batch_size=FLAGS.training_batch_size,
      save_interval=FLAGS.save_interval,
      save_path=save_path,
      model_path=FLAGS.model_path,
      publish_interval=FLAGS.publish_interval,
      learning_rate=FLAGS.learning_rate,
      discount=FLAGS.gamma,
      gae_lambda=FLAGS.gae_lambda,
      pg_clip_epsilon=FLAGS.pg_clip_epsilon,
      value_clip_epsilon=FLAGS.value_clip_epsilon,
      critic_coef=FLAGS.critic_coef,
      entropy_coef=FLAGS.entropy_coef,
    )
    program.add_node(learner_node)
  return program


def main(_):
  """Main function."""
  program = make_lp_program()
  local_resources = {
    "actor":
      PythonProcess(
        env={
          "CUDA_VISIBLE_DEVICES": "",
          "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
        }
      ),
    "learner":
      PythonProcess(
        env={
          "CUDA_VISIBLE_DEVICES": "0",
          "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
        }
      ),
    "predictor":
      PythonProcess(
        env={
          "CUDA_VISIBLE_DEVICES": "0",
          "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
        }
      ),
    "buffer":
      PythonProcess(
        env={
          "CUDA_VISIBLE_DEVICES": "",
          "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
        }
      ),
  }
  lp.launch(program, local_resources=local_resources, terminal="tmux_session")


if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  app.run(main)

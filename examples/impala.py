"""Impala example."""
import os
from typing import Any

import envpool
import launchpad as lp
from absl import app, flags, logging
from launchpad.nodes.python.local_multi_processing import PythonProcess

from moss.actor.base import BaseActor
from moss.agent.base import BaseAgent
from moss.buffer.queue import QueueBuffer
from moss.learner.impala import ImpalaLearner
from moss.network.base import SimpleNet
from moss.predictor.base import BasePredictor
from moss.types import Environment
from moss.utils.loggers import experiment_logger_factory
from moss.utils.paths import get_unique_id

flags.DEFINE_string("task_id", "Pong-v5", "Task name.")
flags.DEFINE_integer("stack_num", 1, "Stack nums.")
flags.DEFINE_bool(
  "use_orthogonal", True, "Use orthogonal to initialization params weight."
)
flags.DEFINE_integer("num_envs", 32, "Num of envs.")
flags.DEFINE_integer("num_threads", 6, "Num threads of envs.")
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
flags.DEFINE_float("gae_lambda", 0.95, "Gae lambda.")
flags.DEFINE_float("rho_clip", 0.9, "Clip threshold for importance ratios.")
flags.DEFINE_float(
  "pg_rho_clip", 0.9, "Clip threshold for policy gradient importance ratios."
)
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
  use_orthogonal = FLAGS.use_orthogonal
  stack_num = FLAGS.stack_num
  num_envs = FLAGS.num_envs
  num_threads = FLAGS.num_threads
  dummy_env: Environment = envpool.make_dm(
    task_id, stack_num=stack_num, num_envs=1
  )
  obs_sepc: Any = dummy_env.observation_spec()
  action_sepc: Any = dummy_env.action_spec()

  logging.info(f"Task id: {task_id}")
  logging.info(f"Observation shape: {obs_sepc.obs.shape}")
  logging.info(f"Action space: {action_sepc.num_values}")

  def network_maker() -> SimpleNet:
    """Network maker."""
    return SimpleNet(obs_sepc, action_sepc, use_orthogonal)

  def env_maker() -> Environment:
    """Env maker."""
    return envpool.make_dm(
      task_id, stack_num=stack_num, num_envs=num_envs, num_threads=num_threads
    )

  def agent_maker() -> BaseAgent:
    """Agent maker."""
    return BaseAgent()

  logger_fn = experiment_logger_factory(
    project=task_id, uid=exp_uid, time_delta=2.0, print_fn=logging.info
  )

  program = lp.Program("impala")
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
        network_maker,
        logger_fn,
      )
      predictor = program.add_node(predictor_node)
      predictors.append(predictor)

  with program.group("actor"):
    for i in range(FLAGS.num_actors):
      actor_node = lp.CourierNode(
        BaseActor,
        buffer,
        agent_maker,
        env_maker,
        predictors[i % FLAGS.num_predictors],
        FLAGS.unroll_len,
        logger_fn,
      )
      program.add_node(actor_node)

  with program.group("learner"):
    save_path = os.path.join("checkpoints", task_id, exp_uid)
    learner_node = lp.CourierNode(
      ImpalaLearner,
      buffer,
      predictors,
      network_maker,
      logger_fn,
      FLAGS.training_batch_size,
      FLAGS.save_interval,
      save_path,
      FLAGS.model_path,
      FLAGS.publish_interval,
      FLAGS.learning_rate,
      FLAGS.gamma,
      FLAGS.gae_lambda,
      FLAGS.rho_clip,
      FLAGS.pg_rho_clip,
      FLAGS.critic_coef,
      FLAGS.entropy_coef,
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

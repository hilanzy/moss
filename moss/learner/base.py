"""Base learner."""
import abc
import os
import pickle
import time
from functools import partial
from typing import Callable, List, Optional, Tuple

import jax
import optax
from absl import logging

from moss.core import Buffer, Learner, Network, Predictor
from moss.types import Array, LoggingData, OptState, Params, Transition
from moss.utils.loggers import Logger


class BaseLearner(Learner):
  """Base learner."""

  def __init__(
    self,
    buffer: Buffer,
    predictors: List[Predictor],
    network_maker: Callable[[], Network],
    logger_fn: Callable[..., Logger],
    batch_size: int,
    save_interval: int,
    save_path: str,
    model_path: Optional[str] = None,
    publish_interval: int = 1,
    learning_rate: float = 5e-4,
    seed: int = 42,
  ) -> None:
    """Init."""
    self._buffer = buffer
    self._predictors = predictors
    self._network = network_maker()
    self._batch_size = batch_size
    self._logger = logger_fn(label="Learner")
    if model_path is not None:
      self._params = self._load_model(model_path)
    else:
      self._params = self._init_params(seed)
    self._publish_params(self._params)
    self._optmizer = optax.rmsprop(learning_rate, decay=0.99, eps=1e-7)
    self._opt_state = self._optmizer.init(self._params)
    self._save_intelval = save_interval
    self._save_fn = partial(self._save_model, save_path=save_path)
    self._publish_interval = publish_interval
    logging.info(jax.devices())

  def _init_params(self, seed: int) -> Params:
    """Init params and update to predictor."""
    rng = jax.random.PRNGKey(seed)
    params = self._network.init_params(rng)
    return params

  def _publish_params(self, params: Params) -> None:
    """Publish params to predictor."""
    for predictor in self._predictors:
      predictor.update_params(params)

  @abc.abstractmethod
  def _loss(self, params: Params, data: Transition) -> Tuple[Array, LoggingData]:
    """Loss function."""

  @partial(jax.jit, static_argnums=0)
  def _train_step(self, params: Params, opt_state: OptState,
                  data: Transition) -> Tuple[Params, OptState, LoggingData]:
    """Training step."""
    (_, metrics), grads = jax.value_and_grad(
      self._loss, has_aux=True
    )(params, data)
    updates, opt_state = self._optmizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, metrics

  def _load_model(self, model_path: str) -> Params:
    """Load model."""
    with open(model_path, mode="rb") as f:
      params: Params = pickle.load(f)
    return params

  def _save_model(self, save_path: str, params: Params) -> None:
    """Save model."""
    os.makedirs(save_path, exist_ok=True)
    suffix = int(time.time())
    save_file = os.path.join(save_path, f"model-{suffix}.pkl")
    params_pkl = pickle.dumps(params)
    with open(save_file, mode="wb") as f:
      f.write(params_pkl)

  def run(self) -> None:
    """Run learner."""
    train_steps = 0
    while True:
      logs = {}
      start_sample_time = time.time()
      tarinig_data = self._buffer.sample(self._batch_size)
      sample_data_time = time.time() - start_sample_time

      start_training_time = time.time()
      self._params, self._opt_state, metrics = self._train_step(
        self._params, self._opt_state, tarinig_data
      )
      train_steps += 1
      training_step_time = time.time() - start_training_time

      publish_params_start = time.time()
      if train_steps % self._publish_interval == 0:
        self._publish_params(self._params)
      publish_params_time = time.time() - publish_params_start

      if train_steps % self._save_intelval == 0:
        self._save_fn(params=self._params)

      logs.update(
        {
          "time/sample data": sample_data_time,
          "time/training step": training_step_time,
          "time/publish params": publish_params_time,
          "train steps": train_steps,
          **metrics,
        }
      )
      self._logger.write(logs)

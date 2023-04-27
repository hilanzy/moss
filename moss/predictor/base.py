"""Base predictor."""
import queue
import threading
import time
from concurrent.futures import Future
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from absl import logging

from moss.core import Network, Params, Predictor
from moss.types import AgentState, Array, KeyArray, NetOutput
from moss.utils.loggers import Logger


class BasePredictor(Predictor):
  """Base predictor."""

  def __init__(
    self,
    batch_size: int,
    network_maker: Callable[[], Network],
    logger_fn: Callable[..., Logger],
    seed: int = 42,
  ) -> None:
    """Init.

    Args:
      batch_size: Predict batch size.
      network_maker: Network maker function.
      logger_fn: Logger function.
      seed: random seed.
    """
    self._batch_size = batch_size
    self._network = network_maker()
    self._logger = logger_fn(label="Predictor")
    self._rng = jax.random.PRNGKey(seed)
    self._requests: queue.Queue[Tuple[Any, Future]] = queue.Queue()
    self._results: Dict[int, Future] = {}
    self._resp_id: int = 0
    self._params: Optional[Params] = None
    self._params_mutex = threading.Lock()
    self._params_initialized = threading.Condition(self._params_mutex)
    self._inference_mutex = threading.Lock()
    logging.info(jax.devices())

  @partial(jax.jit, static_argnums=0)
  def _forward(self, params: Params, state: AgentState,
               rng: KeyArray) -> Tuple[Array, NetOutput]:
    """Forward."""
    action, net_output = self._network.forward(params, state, rng)
    return action, net_output

  def _batch_request(self) -> Tuple[Array, List[Future]]:
    """Get batch request data."""
    batch_obs: Any = []
    futures: List = []
    while len(batch_obs) < self._batch_size:
      try:
        # The function of timeout is to ensure that there
        # is at least one vaild data in batch_obs.
        timetout = 0.05 if len(batch_obs) > 0 else None
        request = self._requests.get(timeout=timetout)
        obs, future = request
        batch_obs.append(obs)
        futures.append(future)
      except queue.Empty:
        logging.info("Get batch request timeout.")
        padding_len = self._batch_size - len(batch_obs)
        padding = jnp.zeros_like(batch_obs[0])
        for _ in range(padding_len):
          batch_obs.append(padding)
        break
    batch_obs = jnp.stack(batch_obs)
    return batch_obs, futures

  def update_params(self, params: Params) -> None:
    """Update params."""
    with self._params_mutex:
      if self._params is None:
        self._params = params
        self._params_initialized.notify_all()
      else:
        self._params = params

  def inference(self, state: AgentState) -> int:
    """Inference."""
    with self._inference_mutex:
      self._resp_id += 1
      resp_id = self._resp_id
    future: Future = Future()
    self._results[resp_id] = future
    self._requests.put((state, future))
    return resp_id

  def result(self, id: int) -> Any:
    """Get result async."""
    future = self._results.pop(id)
    result = future.result()
    return result

  def run(self) -> None:
    """Run predictor."""
    with self._params_initialized:
      if self._params is None:
        self._params_initialized.wait()
    rng = self._rng
    while True:
      get_batch_req_start = time.time()
      batch_obs, futures = self._batch_request()
      get_batch_req_time = time.time() - get_batch_req_start

      forward_start = time.time()
      rng, subrng = jax.random.split(rng)
      action, net_output = self._forward(self._params, batch_obs, subrng)
      (action, net_output) = jax.device_get((action, net_output))
      forward_time = time.time() - forward_start

      for i, future in enumerate(futures):
        future.set_result((action[i], net_output.policy_logits[i]))

      metrics = {
        "time/get batch": get_batch_req_time,
        "time/batch forward": forward_time,
      }
      self._logger.write(metrics)

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

from moss.core import Params, Predictor
from moss.network import Network
from moss.types import AgentState, Array, KeyArray, NetOutput, RNNState
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
    self._requests: queue.Queue[Tuple[AgentState, RNNState,
                                      Future]] = queue.Queue()
    self._results: Dict[int, Future] = {}
    self._resp_id: int = 0
    self._params: Optional[Params] = None
    self._params_mutex = threading.Lock()
    self._params_initialized = threading.Condition(self._params_mutex)
    self._inference_mutex = threading.Lock()
    logging.info(jax.devices())

  @partial(jax.jit, static_argnums=0)
  def _forward(
    self, params: Params, state: AgentState, rnn_state: RNNState, rng: KeyArray
  ) -> Tuple[Dict[str, Array], NetOutput]:
    """Forward."""
    action, net_output = self._network.forward(
      params, state, rnn_state, rng, False
    )
    return action, net_output

  def _batch_request(self) -> Tuple[AgentState, RNNState, List[Future]]:
    """Get batch request data."""
    state_list: List[Any] = []
    rnn_state_list: List[Any] = []
    futures: List[Any] = []
    while len(state_list) < self._batch_size:
      try:
        # The function of timeout is to ensure that there
        # is at least one vaild data in state_list.
        timetout = 0.05 if len(state_list) > 0 else None
        request = self._requests.get(timeout=timetout)
        state, rnn_state, future = request
        state_list.append(state)
        rnn_state_list.append(rnn_state)
        futures.append(future)
      except queue.Empty:
        logging.info("Get batch request timeout.")
        padding_len = self._batch_size - len(state_list)
        state_padding = jax.tree_util.tree_map(
          lambda x: jnp.zeros_like(x), state_list[0]
        )
        rnn_state_padding = jax.tree_util.tree_map(
          lambda x: jnp.zeros_like(x), rnn_state_list[0]
        )
        for _ in range(padding_len):
          state_list.append(state_padding)
          rnn_state_list.append(rnn_state_padding)
        break
    batch_state = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *state_list)
    batch_rnn_state = jax.tree_util.tree_map(
      lambda *x: jnp.stack(x), *rnn_state_list
    )
    return batch_state, batch_rnn_state, futures

  def initial_state(self, batch_size: Optional[int]) -> Any:
    """Get initial state."""
    return self._network.initial_state(batch_size)

  def update_params(self, params: Params) -> None:
    """Update params by Learner."""
    with self._params_mutex:
      if self._params is None:
        self._params = params
        self._params_initialized.notify_all()
      else:
        self._params = params

  def inference(self, state: AgentState, rnn_state: RNNState) -> int:
    """Inference.

    Args:
      state: processed by Agent, and input to Network.
      rnn_state: the last rnn state input to Network.
    """
    with self._inference_mutex:
      self._resp_id += 1
      resp_id = self._resp_id
    future: Future = Future()
    self._results[resp_id] = future
    self._requests.put((state, rnn_state, future))
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
      batch_state, batch_rnn_state, futures = self._batch_request()
      get_batch_req_time = time.time() - get_batch_req_start

      forward_start = time.time()
      rng, sub_rng = jax.random.split(rng)
      action, net_output = self._forward(
        self._params, batch_state, batch_rnn_state, sub_rng
      )
      (action, net_output) = jax.device_get((action, net_output))
      forward_time = time.time() - forward_start

      for i, future in enumerate(futures):
        action_i = jax.tree_map(lambda x: x[i], action)  # noqa: B023
        net_output_i = jax.tree_map(lambda x: x[i], net_output)  # noqa: B023
        result = (
          action_i, net_output_i.policy_logits, net_output_i.value,
          net_output_i.rnn_state
        )
        future.set_result(result)

      metrics = {
        "time/get batch": get_batch_req_time,
        "time/batch forward": forward_time,
      }
      self._logger.write(metrics)

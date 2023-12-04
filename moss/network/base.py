"""Base network."""
import abc
from typing import Any, Callable, Dict, Optional, Tuple

import haiku as hk
import jax.numpy as jnp
import tree

from moss.network.action import ActionSpec
from moss.network.feature import FeatureSpec
from moss.types import AgentState, Array, KeyArray, NetOutput, Params, RNNState


class Network(abc.ABC):
  """Neural network interface."""

  @property
  @abc.abstractmethod
  def action_spec(self) -> ActionSpec:
    """Action spec."""

  @abc.abstractmethod
  def initial_params(self, rng: KeyArray) -> Params:
    """Init network's params."""

  @abc.abstractmethod
  def initial_state(self, batch_size: Optional[int]) -> Any:
    """Constructs an initial state for rnn core."""

  @abc.abstractmethod
  def forward(
    self, params: Params, state: AgentState, rnn_state: RNNState, rng: KeyArray,
    training: bool
  ) -> Tuple[Dict[str, Array], NetOutput]:
    """Network forward."""


class CommonModule(hk.Module):
  """Common haiku module."""

  def __init__(
    self,
    feature_spec: FeatureSpec,
    action_spec: ActionSpec,
    torso_net_maker: Callable[[], Any],
    value_net_maker: Callable[[], Any],
  ) -> None:
    """Init."""
    super().__init__("common_module")
    self._feature_spec = feature_spec
    self._feature_encoder = {
      name: (feature_set.process, feature_set.encoder_net_maker())
      for name, feature_set in feature_spec.feature_sets.items()
    }
    self._action_spec = action_spec
    self._torso_net = torso_net_maker()
    self._value_net = value_net_maker()

  def __call__(
    self, features: Dict, rnn_state: RNNState, training: bool
  ) -> NetOutput:
    """Call."""
    embeddings = {}
    for name, feature_encoder in self._feature_encoder.items():
      feature = features[name]
      processor, encoder = feature_encoder
      if training:
        batch_encoder_apply = hk.BatchApply(encoder)
        batch_process_apply = hk.BatchApply(processor)
        embedding = batch_encoder_apply(batch_process_apply(feature))
      else:
        embedding = encoder(processor(feature))
      embeddings[name] = embedding

    if training:
      if isinstance(self._torso_net, hk.RNNCore):
        torso_out, rnn_state = hk.dynamic_unroll(
          self._torso_net, embeddings, rnn_state
        )
      else:
        batch_torso_apply = hk.BatchApply(self._torso_net)
        torso_out, rnn_state = batch_torso_apply(embeddings, rnn_state)
    else:
      torso_out, rnn_state = self._torso_net(embeddings, rnn_state)

    policy_logits = {}
    for name, action in self._action_spec.actions.items():
      action_mask = features.get(action.mask_on)
      if training:
        batch_policy_apply = hk.BatchApply(action.policy_net)
        policy_logits[name] = batch_policy_apply(torso_out, action_mask)
      else:
        policy_logits[name] = action.policy_net(torso_out, action_mask)

    if training:
      batch_value_apply = hk.BatchApply(self._value_net)
      value = batch_value_apply(torso_out)
    else:
      value = self._value_net(torso_out)

    return NetOutput(policy_logits, value, rnn_state)

  def initial_state(self, batch_size: Optional[int]) -> RNNState:
    """Constructs an initial state for rnn core."""
    try:
      rnn_state = self._torso_net.initial_state(batch_size)
    except Exception:
      rnn_state = None
    return rnn_state


class CommonNet(Network):
  """Common network."""

  def __init__(
    self,
    feature_spec: FeatureSpec,
    action_spec: ActionSpec,
    torso_net_maker: Callable[[], Any],
    value_net_maker: Callable[[], Any],
  ) -> None:
    """Init."""
    self._feature_spec = feature_spec
    self._action_spec = action_spec
    self._net = hk.without_apply_rng(
      hk.transform(
        lambda *args, **kwargs: CommonModule(
          feature_spec, action_spec, torso_net_maker, value_net_maker
        )(*args, **kwargs)
      )
    )
    self._initial_state = hk.without_apply_rng(
      hk.transform(
        lambda x: CommonModule(
          feature_spec, action_spec, torso_net_maker, value_net_maker
        ).initial_state(x)
      )
    )

  @property
  def action_spec(self) -> ActionSpec:
    """Action spec."""
    return self._action_spec

  def initial_params(self, rng: KeyArray) -> Params:
    """Init network's params."""
    dummy_inputs = self._feature_spec.generate_value()
    dummy_inputs = tree.map_structure(
      lambda x: jnp.expand_dims(x, 0), dummy_inputs
    )  # shape: [B, ...]
    init_rnn_state = self.initial_state(1)
    params = self._net.init(rng, dummy_inputs, init_rnn_state, False)
    return params

  def initial_state(self, batch_size: Optional[int]) -> RNNState:
    """Constructs an initial state for rnn core."""
    return self._initial_state.apply(None, batch_size)

  def forward(
    self, params: Params, state: AgentState, rnn_state: RNNState, rng: KeyArray,
    training: bool
  ) -> Tuple[Dict[str, Array], NetOutput]:
    """Network forward."""
    net_output = self._net.apply(params, state, rnn_state, training)
    actions = self._action_spec.sample(rng, net_output.policy_logits)
    return actions, net_output

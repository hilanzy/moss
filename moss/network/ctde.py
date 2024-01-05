"""Centealized training and decentralized execution(CTDE) network."""
from typing import Dict, Optional, Tuple

import haiku as hk
import jax.numpy as jnp
import tree

from moss.network import Network
from moss.network.action import ActionSpec
from moss.network.base import Module, RNNModule
from moss.network.feature import FeatureSpec
from moss.network.keys import AGENT_STATE, GLOBAL_STATE, MASK
from moss.types import Array, KeyArray, NetOutput, Params, RNNState


class CTDEModule(hk.Module):
  """CTDE haiku module."""

  def __init__(
    self,
    feature_spec: FeatureSpec,
    global_feature_spec: FeatureSpec,
    action_spec: ActionSpec,
    torso_net: Module,
    value_net: Module,
  ) -> None:
    """Init."""
    super().__init__("CTDE_module")
    self._feature_spec = feature_spec
    self._feature_encoder = {
      name: (feature_set.process, feature_set.encoder_net)
      for name, feature_set in feature_spec.feature_sets.items()
    }
    self._golbal_feature_encoder = {
      name: (feature_set.process, feature_set.encoder_net)
      for name, feature_set in global_feature_spec.feature_sets.items()
    }
    self._action_spec = action_spec
    self._torso_net = torso_net
    self._value_net = value_net

  def __call__(
    self, input_dict: Dict, rnn_state: RNNState, training: bool
  ) -> NetOutput:
    """Call."""
    # Actor network.
    agent_embeddings = {}
    agent_features = input_dict[AGENT_STATE]
    for name, feature_encoder in self._feature_encoder.items():
      feature = agent_features[name]
      processor, encoder = feature_encoder
      if training:
        batch_encoder_apply = hk.BatchApply(encoder)
        batch_process_apply = hk.BatchApply(processor)
        embedding = batch_encoder_apply(batch_process_apply(feature))
      else:
        embedding = encoder(processor(feature))
      agent_embeddings[name] = embedding

    if training:
      if isinstance(self._torso_net, RNNModule):
        torso_out, rnn_state = hk.dynamic_unroll(
          self._torso_net, agent_embeddings, rnn_state
        )
      else:
        batch_torso_apply = hk.BatchApply(self._torso_net)
        torso_out, rnn_state = batch_torso_apply(agent_embeddings, rnn_state)
    else:
      torso_out, rnn_state = self._torso_net(agent_embeddings, rnn_state)

    policy_logits = {}
    mask = input_dict.get(MASK, {})
    for name, action in self._action_spec.actions.items():
      action_mask = mask.get(action.name)
      if training:
        batch_policy_apply = hk.BatchApply(action.policy_net)
        policy_logits[name] = batch_policy_apply(torso_out, action_mask)
      else:
        policy_logits[name] = action.policy_net(torso_out, action_mask)

    # Critic network.
    global_embeddings = {}
    global_features = input_dict[GLOBAL_STATE]
    for name, feature_encoder in self._golbal_feature_encoder.items():
      feature = global_features[name]
      processor, encoder = feature_encoder
      if training:
        batch_encoder_apply = hk.BatchApply(encoder)
        batch_process_apply = hk.BatchApply(processor)
        embedding = batch_encoder_apply(batch_process_apply(feature))
      else:
        embedding = encoder(processor(feature))
      global_embeddings[name] = embedding
    global_embedding = jnp.concatenate(list(global_embeddings.values()), axis=-1)

    if training:
      batch_value_apply = hk.BatchApply(self._value_net)
      value = batch_value_apply(global_embedding)
    else:
      value = self._value_net(global_embedding)

    return NetOutput(policy_logits, value, rnn_state)

  def initial_state(self, batch_size: Optional[int]) -> RNNState:
    """Constructs an initial state for rnn core."""
    if isinstance(self._torso_net, RNNModule):
      rnn_state = self._torso_net.initial_state(batch_size)
    else:
      rnn_state = None
    return rnn_state


class CTDENet(Network):
  """CTDE network."""

  def __init__(
    self,
    feature_spec: FeatureSpec,
    global_feature_spec: FeatureSpec,
    action_spec: ActionSpec,
    torso_net: Module,
    value_net: Module,
  ) -> None:
    """Init."""
    self._feature_spec = feature_spec
    self._global_feature_spec = global_feature_spec
    self._action_spec = action_spec
    self._net = hk.without_apply_rng(
      hk.transform(
        lambda *args, **kwargs: CTDEModule(
          feature_spec, global_feature_spec, action_spec, torso_net, value_net
        )(*args, **kwargs)
      )
    )
    self._initial_state = hk.without_apply_rng(
      hk.transform(
        lambda x: CTDEModule(
          feature_spec, global_feature_spec, action_spec, torso_net, value_net
        ).initial_state(x)
      )
    )

  @property
  def action_spec(self) -> ActionSpec:
    """Action spec."""
    return self._action_spec

  def initial_params(self, rng: KeyArray) -> Params:
    """Init network's params."""
    dummy_agent_state = self._feature_spec.generate_value()
    dummy_global_state = self._global_feature_spec.generate_value()
    dummy_inputs = {
      AGENT_STATE: dummy_agent_state,
      GLOBAL_STATE: dummy_global_state,
    }
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
    self, params: Params, input_dict: Dict, rnn_state: RNNState, rng: KeyArray,
    training: bool
  ) -> Tuple[Dict[str, Array], NetOutput]:
    """Network forward."""
    net_output = self._net.apply(params, input_dict, rnn_state, training)
    actions = self._action_spec.sample(rng, net_output.policy_logits)
    return actions, net_output

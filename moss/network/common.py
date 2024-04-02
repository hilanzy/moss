"""Common network."""
from typing import Dict, Tuple

import flax.linen as nn
import jax.numpy as jnp
import tree

from moss.network import Network
from moss.network.action import ActionSpec
from moss.network.feature import FeatureSpec
from moss.network.keys import AGENT_STATE, MASK
from moss.network.torso import Torso
from moss.network.utils import BatchApply
from moss.network.value import Value
from moss.types import Array, KeyArray, NetOutput, Params, RNNState


class CommonModule(nn.Module):
  """Common flax module."""
  feature_spec: FeatureSpec
  action_spec: ActionSpec
  torso: Torso
  value: Value
  name = "common_module"

  @nn.compact
  def __call__(
    self, input_dict: Dict, rnn_state: RNNState, training: bool
  ) -> NetOutput:
    """Call."""
    feature_encoders = {
      name: (feature_set.process, feature_set.encoder)
      for name, feature_set in self.feature_spec.feature_sets.items()
    }
    embeddings = {}
    features = input_dict[AGENT_STATE]
    for name, feature_encoder in feature_encoders.items():
      feature = features[name]
      processor, encoder = feature_encoder
      if training:
        batch_encoder_apply = BatchApply(encoder)
        batch_process_apply = BatchApply(processor)
        embedding = batch_encoder_apply(batch_process_apply(feature))
      else:
        embedding = encoder(processor(feature))
      embeddings[name] = embedding

    embedding = jnp.concatenate(list(embeddings.values()), axis=-1)
    torso_out, rnn_state = self.torso.aggregator(embedding, rnn_state, training)

    policy_logits = {}
    mask = input_dict.get(MASK, {})
    for name, action in self.action_spec.actions.items():
      action_mask = mask.get(action.name)
      if training:
        batch_decoder_apply = BatchApply(action.decoder)
        policy_logits[name] = batch_decoder_apply(torso_out, action_mask)
      else:
        policy_logits[name] = action.decoder(torso_out, action_mask)

    if training:
      batch_decoder_apply = BatchApply(self.value.decoder)
      value = batch_decoder_apply(torso_out)
    else:
      value = self.value.decoder(torso_out)

    return NetOutput(policy_logits, value, rnn_state)


class CommonNet(Network):
  """Common network."""

  def __init__(
    self,
    feature_spec: FeatureSpec,
    action_spec: ActionSpec,
    torso: Torso,
    value: Value,
  ) -> None:
    """Init."""
    self._feature_spec = feature_spec
    self._action_spec = action_spec
    self._net = CommonModule(feature_spec, action_spec, torso, value)

  @property
  def action_spec(self) -> ActionSpec:
    """Action spec."""
    return self._action_spec

  def initial_params(self, rng: KeyArray) -> Params:
    """Init network's params."""
    dummy_agent_state = self._feature_spec.generate_value()
    dummy_inputs = {AGENT_STATE: dummy_agent_state}
    dummy_inputs = tree.map_structure(
      lambda x: jnp.expand_dims(x, 0), dummy_inputs
    )  # shape: [B, ...]
    variables = self._net.init(rng, dummy_inputs, None, False)
    return variables["params"]

  def forward(
    self, params: Params, input_dict: Dict, rnn_state: RNNState, rng: KeyArray,
    training: bool
  ) -> Tuple[Dict[str, Array], NetOutput]:
    """Network forward."""
    variables = {"params": params}
    net_output = self._net.apply(variables, input_dict, rnn_state, training)
    policy_logits = net_output.policy_logits  # type: ignore
    actions = self._action_spec.sample(rng, policy_logits)
    return actions, net_output  # type: ignore

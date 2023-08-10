"""Base network."""
from typing import Any, Callable, Dict, List, Tuple

import haiku as hk
import jax.numpy as jnp
import tree

from moss.core import Network
from moss.network.action_spec import ActionSpec
from moss.network.feature_set import CommonFeatureSet
from moss.types import AgentState, Array, KeyArray, NetOutput, Params


class CommonModule(hk.Module):
  """Common haiku module."""

  def __init__(
    self,
    feature_spec: List[CommonFeatureSet],
    action_spec: ActionSpec,
    torso_net_maker: Callable[[], Any],
    value_net_maker: Callable[[], Any],
  ) -> None:
    """Init."""
    super().__init__("common_module")
    self._feature_spec = feature_spec
    self._feature_encoder = {
      sepc.name: (sepc.process, sepc.encoder_net_maker) for sepc in feature_spec
    }
    self._action_spec = action_spec
    self._torso_net_maker = torso_net_maker
    self._value_net_maker = value_net_maker

  def __call__(self, features: Dict) -> Tuple[Array, Array]:
    """Call."""
    embeddings = []
    for name, feature in features.items():
      processor, encoder_net_maker = self._feature_encoder[name]
      encoder_net = encoder_net_maker()
      embedding = encoder_net(processor(feature))
      embeddings.append(embedding)
    embeddings = jnp.concatenate(embeddings, axis=0)

    torso_net = self._torso_net_maker()
    torso_out = torso_net(embeddings)

    # policy logits
    policy_logits = self._action_spec.action_net(torso_out)
    value_net = self._value_net_maker()

    # value
    value = value_net(torso_out)

    return policy_logits, value


class CommonNet(Network):
  """Common network."""

  def __init__(
    self,
    feature_spec: List[CommonFeatureSet],
    action_spec: ActionSpec,
    torso_net_maker: Callable[[], Any],
    value_net_maker: Callable[[], Any],
  ) -> None:
    """Init."""
    self._feature_spec = feature_spec
    self._action_spec = action_spec
    self._net = hk.without_apply_rng(
      hk.transform(
        lambda x: CommonModule(
          feature_spec, action_spec, torso_net_maker, value_net_maker
        )(x)
      )
    )

  @property
  def action_spec(self) -> ActionSpec:
    """Action spec."""
    return self._action_spec

  def init_params(self, rng: KeyArray) -> Params:
    """Init network's params."""
    dummy_inputs = {
      spec.name: spec.generate_value() for spec in self._feature_spec
    }
    dummy_inputs = tree.map_structure(
      lambda x: jnp.expand_dims(x, 0), dummy_inputs
    )
    params = self._net.init(rng, dummy_inputs)
    return params

  def forward(self, params: Params, state: AgentState,
              rng: KeyArray) -> Tuple[Dict[str, Array], NetOutput]:
    """Network forward."""
    policy_logits, value = self._net.apply(params, state)
    actions = self._action_spec.sample(rng, policy_logits)
    return actions, NetOutput(policy_logits, value)

"""Base network."""
import abc
from typing import Any, Callable, Dict, Tuple

import haiku as hk
import jax.numpy as jnp
import tree

from moss.network.action import ActionSpec
from moss.network.feature import FeatureSpec
from moss.types import AgentState, Array, KeyArray, NetOutput, Params


class Network(abc.ABC):
  """Neural network interface."""

  @property
  @abc.abstractmethod
  def action_spec(self) -> ActionSpec:
    """Action spec."""

  @abc.abstractmethod
  def init_params(self, rng: KeyArray) -> Params:
    """Init network's params."""

  @abc.abstractmethod
  def forward(self, params: Params, state: AgentState,
              rng: KeyArray) -> Tuple[Dict[str, Array], NetOutput]:
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
      name: (feature_set.process, feature_set.encoder_net_maker)
      for name, feature_set in feature_spec.feature_sets.items()
    }
    self._action_spec = action_spec
    self._torso_net_maker = torso_net_maker
    self._value_net_maker = value_net_maker

  def __call__(self, features: Dict) -> Tuple[Dict[str, Array], Array]:
    """Call."""
    embeddings = {}
    for name, feature in features.items():
      processor, encoder_net_maker = self._feature_encoder[name]
      encoder_net = encoder_net_maker()
      embedding = encoder_net(processor(feature))
      embeddings[name] = embedding

    torso_net = self._torso_net_maker()
    torso_out = torso_net(embeddings)

    # policy logits
    policy_logits = {}
    for name, action in self._action_spec.actions.items():
      action_mask = features.get(action.mask_on)
      policy_logits[name] = action.policy_net(torso_out, action_mask)

    # value
    value_net = self._value_net_maker()
    value = value_net(torso_out)

    return policy_logits, value


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
    dummy_inputs = self._feature_spec.generate_value()
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

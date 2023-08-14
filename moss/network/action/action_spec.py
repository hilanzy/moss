"""Action spec."""
import collections
from typing import Any, Dict

from distrax import DistributionLike

from moss.network.action.base import Action
from moss.types import Array, KeyArray

ActionDistribution = collections.namedtuple(
  "ActionDistribution", ["log_prob", "prob", "entropy"]
)


class ActionSpec(object):
  """Action spec."""

  def __init__(self, actions: Dict[str, Action]) -> None:
    """Init."""
    self._actions = actions
    self._spec = {name: action.spec for name, action in actions.items()}

  @property
  def actions(self) -> Dict[str, Action]:
    """Actions."""
    return self._actions

  @property
  def action_spec(self) -> Any:
    """Action specs."""
    return self._spec

  def distribution(self, logits: Dict[str, Any]) -> DistributionLike:
    """Action distribution."""

    def log_prob(value: Dict[str, Array]) -> Array:
      """See `Distribution.log_prob`."""
      prob: Array = 0.0  # type: ignore
      for name, action in self._actions.items():
        prob += action.distribution(logits[name]).log_prob(value[name])
      return prob

    def prob(value: Dict[str, Array]) -> Array:
      """See `Distribution.prob`."""
      prob: Array = 1.0  # type: ignore
      for name, action in self._actions.items():
        prob *= action.distribution(logits[name]).prob(value[name])
      return prob

    def entropy() -> Array:
      """See `Distribution.entropy`."""
      ent: Array = 0.0  # type: ignore
      for name, action in self._actions.items():
        ent += action.distribution(logits[name]).entropy()
      return ent

    return ActionDistribution(log_prob, prob, entropy)

  def sample(self, rng: KeyArray, logits: Dict[str, Any]) -> Any:
    """Sample actions."""
    actions = {
      name: action.sample(rng, logits[name])
      for name, action in self._actions.items()
    }
    return actions

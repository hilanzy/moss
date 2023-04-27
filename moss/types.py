"""Types."""
import collections
from typing import Any, Dict, List, NamedTuple, Union

import haiku as hk
import jax
import numpy as np
import optax
from dm_env import Environment, StepType  # noqa: F401
from jax.random import KeyArray  # noqa: F401

Array = Union[np.ndarray, jax.Array]
Params = Union[hk.Params, optax.Params]
OptState = optax.OptState
LoggingData = Dict[str, Any]

Observation = Any
AgentState = Any
Action = Any
Reward = Any
History = Any


class Transition(NamedTuple):
  """Transtion."""
  step_type: StepType
  state: AgentState
  action: Action
  reward: Reward
  policy_logits: Array


Trajectory = Union[Transition, List[Transition]]
NetOutput = collections.namedtuple("NetOutput", ["policy_logits", "value"])

"""Types."""
import collections
from typing import Any, Dict, List, NamedTuple, Union

import haiku as hk
import jax
import numpy as np
import optax
from chex import Numeric  # noqa: F401
from dm_env import Environment, StepType, TimeStep  # noqa: F401
from dm_env.specs import Array as SpecArray  # noqa: F401
from dm_env.specs import BoundedArray, DiscreteArray  # noqa: F401
from jax.random import KeyArray  # noqa: F401

Array = Union[np.ndarray, jax.Array]
Params = Union[hk.Params, optax.Params]
OptState = optax.OptState
LoggingData = Dict[str, Any]

Observation = Any
AgentState = Any
RNNState = Any
Action = Any
Reward = Any
History = Any


class Transition(NamedTuple):
  """Transtion."""
  step_type: StepType
  input_dict: Dict
  action: Action
  rnn_state: RNNState
  reward: Reward
  policy_logits: Dict[str, Array]
  behaviour_value: Array


Trajectory = Union[Transition, List[Transition]]
NetOutput = collections.namedtuple(
  "NetOutput", ["policy_logits", "value", "rnn_state"]
)

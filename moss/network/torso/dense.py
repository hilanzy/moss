"""Dense torso network."""
from typing import Any, List, Optional, Tuple

import flax.linen as nn
import jax

from moss.network.torso.base import Torso
from moss.types import Array, RNNState


class DenseTorso(Torso):
  """Dense torso network."""

  def __init__(
    self,
    hidden_sizes: List[int],
    name: Optional[str] = None,
    use_orthogonal: bool = True
  ):
    """Dense torso init."""
    self._name = name or "dense_torso"
    self._hidden_sizes = hidden_sizes
    self._use_orthogonal = use_orthogonal

  def aggregator(
    self,
    inputs: Array,
    rnn_state: RNNState,
    training: bool = False
  ) -> Tuple[Array, RNNState]:
    """Aggregator.

    Args:
      inputs: input features.
      rnn_state: rnn state, ignore for this class.
      training: whether is trainig mode, ignore for this class.

    Returns:
      torso_out: torso aggregator output.
      rnn_state: rnn state, ignore for this class.
    """
    del training
    layers: List[Any] = []
    init_kwargs = {}
    if self._use_orthogonal:
      init_kwargs["kernel_init"] = nn.initializers.orthogonal()
    for hidden_size in self._hidden_sizes:
      layers.append(nn.Dense(hidden_size, **init_kwargs))
      layers.append(jax.nn.relu)
    torso_net = nn.Sequential(layers)
    torso_out = torso_net(inputs)
    return torso_out, rnn_state

  @property
  def name(self) -> str:
    """Get torso name."""
    return self._name

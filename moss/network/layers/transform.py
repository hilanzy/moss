# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transformer module.

Glossary of shapes:
- B: Batch size.
- T: Sequence length.
- D: Model embedding size.
- H: Number of attention heads.
"""
import dataclasses
from typing import Optional

import haiku as hk
import jax
import numpy as np


def _layer_norm(x: jax.Array) -> jax.Array:
  """Applies a unique LayerNorm to `x` with default settings."""
  ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
  return ln(x)


@dataclasses.dataclass
class Transformer(hk.Module):
  """A transformer stack."""

  num_heads: int  # Number of attention heads.
  num_layers: int  # Number of transformer (attention + MLP) layers to stack.
  attn_size: int  # Size of the attention (key, query, value) vectors.
  dropout_rate: float  # Probability with which to apply dropout.
  widening_factor: int = 4  # Factor by which the MLP hidden layer widens.
  name: Optional[str] = None  # Optional identifier for the module.

  def __call__(
    self,
    embeddings: jax.Array,  # [B, T, D]
    mask: jax.Array,  # [B, T]
  ) -> jax.Array:  # [B, T, D]
    """Transforms input embedding sequences to output embedding sequences."""
    initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
    _, seq_len, model_size = embeddings.shape

    # Compute causal mask for autoregressive sequence modelling.
    mask = mask[:, None, None, :]  # [B, H=1, T'=1, T]
    causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))  # [B=1, H=1, T, T]
    mask = mask * causal_mask  # [B, H=1, T, T]

    h = embeddings
    for _ in range(self.num_layers):
      # First the attention block.
      attn_block = hk.MultiHeadAttention(
        num_heads=self.num_heads,
        key_size=self.attn_size,
        model_size=model_size,
        w_init=initializer,
      )
      h_norm = _layer_norm(h)
      h_attn = attn_block(h_norm, h_norm, h_norm, mask=mask)
      h_attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_attn)
      h = h + h_attn

      # Then the dense block.
      dense_block = hk.Sequential(
        [
          hk.Linear(self.widening_factor * model_size, w_init=initializer),
          jax.nn.gelu,
          hk.Linear(model_size, w_init=initializer),
        ]
      )
      h_norm = _layer_norm(h)
      h_dense = dense_block(h_norm)
      h_dense = hk.dropout(hk.next_rng_key(), self.dropout_rate, h_dense)
      h = h + h_dense

    return _layer_norm(h)

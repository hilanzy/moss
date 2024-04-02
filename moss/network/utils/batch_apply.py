# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Batch apply utils."""
# flake8: noqa
import jax
import jax.numpy as jnp
import numpy as np


def ndim_at_least(x, num_dims):  # type: ignore
  """Ndim at least."""
  if not isinstance(x, jnp.ndarray):
    x = jnp.asarray(x)
  return x.ndim >= num_dims


def arbitrary_mergeable_leaf(min_num_dims, args, kwargs):  # type: ignore
  """Arbitrary mergeable leaf."""
  for a in jax.tree_util.tree_leaves(args):
    if ndim_at_least(a, min_num_dims):
      return a
  for k in jax.tree_util.tree_leaves(kwargs):
    if ndim_at_least(k, min_num_dims):
      return k
  # Couldn't find a satisfactory leaf.
  return None


def merge_leading_dims(x, num_dims):  # type: ignore
  """Merge leading dimensions."""
  # Don't merge if there aren't dimensions to merge.
  if not ndim_at_least(x, num_dims):
    return x

  # TODO(tomhennigan) Pass dtype here to account for empty slices.
  new_shape = (np.prod(x.shape[:num_dims]),) + x.shape[num_dims:]
  return jnp.reshape(x, new_shape)


def split_leading_dim(x, to_dim):  # type: ignore
  """Split leading dim."""
  new_shape = to_dim + x.shape[1:]
  return jnp.reshape(x, new_shape)


class BatchApply:
  r"""Temporarily merges leading dimensions of input tensors.

  Merges the leading dimensions of a tensor into a single dimension, runs the
  given callable, then splits the leading dimension of the result to match the
  input.

  Input arrays whose rank is smaller than the number of dimensions to collapse
  are passed unmodified.

  This may be useful for applying a module to each timestep of e.g. a
  ``[Time, Batch, ...]`` array.

  For some ``f``\ s and platforms, this may be more efficient than
  :func:`jax.vmap`, especially when combined with other transformations like
  :func:`jax.grad`.
  """

  def __init__(self, f, num_dims=2):  # type: ignore
    """Constructs a :class:`BatchApply` module.

    Args:
      f: The callable to be applied to the reshaped array.
      num_dims: The number of dimensions to merge.
    """
    self._f = f
    self.num_dims = num_dims

  def __call__(self, *args, **kwargs):  # type: ignore
    """Call."""
    example = arbitrary_mergeable_leaf(self.num_dims, args, kwargs)
    if example is None:
      raise ValueError(
        "BatchApply requires at least one input with ndim >= "
        f"{self.num_dims}."
      )

    merge = lambda x: merge_leading_dims(x, self.num_dims)
    split = lambda x: split_leading_dim(x, example.shape[:self.num_dims])
    args = jax.tree_util.tree_map(merge, args)
    kwargs = jax.tree_util.tree_map(merge, kwargs)
    outputs = self._f(*args, **kwargs)
    return jax.tree_util.tree_map(split, outputs)

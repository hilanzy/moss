"""Vizdoom example utils."""
import collections
from typing import Any, List, Optional, Tuple

import envpool
import numpy as np
import pygame
import tree
from pygame import Surface

from moss.types import Environment, TimeStep


def action_spec_wrapper(env) -> Any:
  """Action spec wrapper.

  This function is to wrapper action space to dm action spec, to avoid
    `action_spec()` bug(maybe).
    See: https://github.com/sail-sg/envpool/issues/264
  """
  ActionSpec = collections.namedtuple("ActionSpec", ["num_values"])
  return ActionSpec(env.spec.action_space.n)


class LocalEnv(Environment):
  """Local environment wrapper for Atari games using EnvPool and Pygame.

  This class is a local environment wrapper that provides an interface
  to interact with Atari games. It is built on top of the EnvPool library,
  which provides efficient environment pooling for reinforcement learning,
  and uses Pygame for rendering the game frames.

  Attributes:
    task_id (str): Identifier for the specific Atari game.
    env (Environment): The EnvPool environment instance for the game logic.
    render (bool): Flag to determine whether to render the game frames.
    fps (int): Frames per second for game rendering.
    scale (int): Scaling factor for rendering the game frames.
    window (Optional[Surface]): Pygame window surface for rendering.
    screen_size (Optional[Tuple[float, float]]): The size of the rendering
      window.
    clock: Pygame clock object for controlling the render loop timing.

  Args:
    task_id (str): Identifier for the specific Atari game.
    seed (int, optional): Random seed for the environment.
    render (bool, optional): Whether to render the game frames.
    fps (int, optional): Frames per second for game rendering.
    scale (int, optional): Scaling factor for rendering the game frames.

  """

  def __init__(
    self,
    task_id: str,
    seed: int = 42,
    render: bool = True,
    fps: int = 60,
    scale: int = 1,
    **kwargs: Any,
  ) -> None:
    """Initializes the local Atari environment with optional rendering.

    Args:
      task_id (str): Identifier for the specific Atari game.
      seed (int): Random seed for the environment.
      render (bool): Whether to render the game frames.
      fps (int): Frames per second for game rendering.
      scale (int): Scaling factor for rendering the game frames.
      kwargs (Any): Any arguments for env.
    """
    self._task_id = task_id
    self._env = envpool.make_dm(
      task_id,
      num_envs=1,
      stack_num=1,
      seed=seed,
      **kwargs,
    )
    self._render_mode = render
    if render:
      # set the same seed to env and render env.
      self._render_env: Environment = envpool.make_dm(
        task_id,
        num_envs=1,
        stack_num=1,
        seed=seed,
        img_height=210,
        img_width=160,
        **kwargs,
      )
      self._fps = fps
      self._scale = scale
      self._window: Optional[Surface] = None
      self._screen_size: Optional[Tuple[float, float]] = None
      self._clock = pygame.time.Clock()

  def _split_batch_timestep(self, batch: TimeStep) -> List[TimeStep]:
    """Splits a batched timestep into a list of individual timesteps.

    Args:
      batch (TimeStep): The batched timestep to split.

    Returns:
      List[TimeStep]: A list of individual timesteps.
    """
    size = batch.step_type.size
    timesteps = [
      tree.map_structure(lambda x: x[i], batch)  # noqa: B023
      for i in range(size)
    ]
    return timesteps

  def reset(self) -> TimeStep:
    """Resets the environment and returns the initial observation.

    Returns:
        TimeStep: The initial timestep after resetting the environment.
    """
    if self._render_mode:
      render_timestep = self._render_env.reset()
      render_timestep = self._split_batch_timestep(render_timestep)[0]
      self.render(render_timestep)
    timestep = self._env.reset()
    timestep = self._split_batch_timestep(timestep)[0]
    return timestep

  def step(self, action) -> TimeStep:
    """Steps the environment and returns the new timestep.

    Args:
      action: The action to take in the environment.

    Returns:
      TimeStep: The timestep resulting from the action.
    """
    if self._render_mode:
      render_timestep = self._render_env.step(action)
      render_timestep = self._split_batch_timestep(render_timestep)[0]
      self.render(render_timestep)
    timestep = self._env.step(action)
    timestep = self._split_batch_timestep(timestep)[0]
    return timestep

  def observation_spec(self) -> Any:
    """Defines the observations provided by the environment.

    May use a subclass of `specs.Array` that specifies additional properties
    such as min and max bounds on the values.

    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    return self._env.observation_spec()

  def action_spec(self) -> Any:
    """Defines the actions that should be provided to `step`.

    May use a subclass of `specs.Array` that specifies additional properties
    such as min and max bounds on the values.

    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    return action_spec_wrapper(self._env)

  def render(self, timestep: TimeStep) -> None:
    """Renders the current game frame based on the latest timestep.

    Args:
      timestep (TimeStep): The timestep to render.
    """
    obs = timestep.observation.obs  # type: ignore
    gray_obs = np.concatenate([obs] * 3, axis=0)
    gray_array = np.transpose(
      gray_obs, axes=(2, 1, 0)
    )  # (height, width, channel)

    if self._screen_size is None:
      width, height = gray_array.shape[:2]
      self._screen_size = (width * self._scale, height * self._scale)

    if self._window is None:
      pygame.init()
      pygame.display.init()
      pygame.display.set_caption(self._task_id)
      self._window = pygame.display.set_mode(self._screen_size)

    surf = pygame.surfarray.make_surface(gray_array)
    surf = pygame.transform.scale(surf, self._screen_size)
    self._window.blit(surf, (0, 0))
    pygame.event.pump()
    self._clock.tick(self._fps)
    pygame.display.flip()

  def close(self) -> None:
    """Releases resources and closes rendering window."""
    self._env.close()
    if self._render_mode:
      self._render_env.close()
      if self._window is not None:
        pygame.display.quit()
        pygame.quit()

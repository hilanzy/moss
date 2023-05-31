"""Atari utils."""
from typing import Any, List, Optional, Tuple

import envpool
import numpy as np
import pygame
import tree
from dm_env import Environment, TimeStep
from pygame import Surface

from moss.env import TimeStep


class LocalEnv(Environment):
  """Atari local env wrapper."""

  def __init__(self, task_id: str, fps: int = 60, scale: int = 10) -> None:
    """Init."""
    self._task_id = task_id
    self._fps = fps
    self._scale = scale
    self._env: Environment = envpool.make_dm(task_id, stack_num=1, num_envs=1)
    self._window: Optional[Surface] = None
    self._screen_size: Optional[Tuple[float, float]] = None
    self._clock = pygame.time.Clock()
    self._last_timestep: Optional[TimeStep] = None

  def _split_batch_timestep(self, batch: TimeStep) -> List[TimeStep]:
    """Split batch timestep by env."""
    size = batch.step_type.size
    timesteps = [
      tree.map_structure(lambda x: x[i], batch)  # noqa: B023
      for i in range(size)
    ]
    return timesteps

  def reset(self) -> Any:
    """Env reset."""
    timestep = self._env.reset()
    self._last_timestep = self._split_batch_timestep(timestep)[0]
    return self._last_timestep

  def step(self, action) -> Any:
    """Env step."""
    timestep = self._env.step(action)
    self._last_timestep = self._split_batch_timestep(timestep)[0]
    return self._last_timestep

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
    return self._env.action_spec()

  def render(self) -> None:
    """Render."""
    obs = self._last_timestep.observation.obs  # type: ignore
    gray_array = np.transpose(obs, axes=(2, 1, 0))  # (height, width, channel)
    rgb_array = np.concatenate([gray_array] * 3, axis=-1)

    if self._screen_size is None:
      width, height = rgb_array.shape[:2]
      self._screen_size = (width * self._scale, height * self._scale)

    if self._window is None:
      pygame.init()
      pygame.display.init()
      pygame.display.set_caption(self._task_id)
      self._window = pygame.display.set_mode(self._screen_size)

    surf = pygame.surfarray.make_surface(rgb_array)
    surf = pygame.transform.scale(surf, self._screen_size)
    self._window.blit(surf, (0, 0))
    pygame.event.pump()
    self._clock.tick(self._fps)
    pygame.display.flip()

  def close(self) -> None:
    """Close the rendering window."""
    super().close()
    if self._window is not None:
      pygame.display.quit()
      pygame.quit()

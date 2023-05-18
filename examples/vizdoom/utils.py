"""Vizdoom example utils."""
import envpool
from dm_env import Environment


def vizdoom_env_maker(
  map_id: str, num_envs: int, cfg_path: str, wad_path: str, **kwargs
) -> Environment:
  """Vizdoom env maker."""
  task_id = "".join([i.capitalize() for i in map_id.split("_")]) + "-v1"
  reward_config = {
    "KILLCOUNT": [20.0, -20.0],
    "HEALTH": [1.0, 0.0],
    "AMMO2": [1.0, -1.0],
  }
  if "battle" in task_id:
    reward_config["HEALTH"] = [1.0, -1.0]
  return envpool.make_dm(
    task_id,
    num_envs=num_envs,
    cfg_path=cfg_path,
    wad_path=wad_path,
    reward_config=reward_config,
    use_combined_action=True,
    max_episode_steps=2625,
    use_inter_area_resize=False,
    **kwargs,
  )

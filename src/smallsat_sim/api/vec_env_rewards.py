"""Public built-in reward names; numerical definitions live with the environment."""
from smallsat_sim.api.registry import register_reward
from smallsat_sim.envs.rewards import full_pose_reward

register_reward("full_pose", full_pose_reward)
register_reward("vec_env/full_pose", full_pose_reward)

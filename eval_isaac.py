# eval_isaac.py
import numpy as np
import os


from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401, F403
from legged_gym.utils import task_registry
import isaacgym  # noqa: F401  # makes sure gym loads before envs

import torch

@torch.no_grad()
def eval_actor_isaac(
    actor: torch.nn.Module,
    task_name: str = "anymal_c_flat",
    num_envs: int = 1,
    n_episodes: int = 10,
    seed: int = 0,
    device: str = "cuda",
    headless: bool = True,
    record: bool = False,
    move_camera: bool = False,
) -> np.ndarray:
    """
    Evaluate a custom actor (e.g. from CQL) in Isaac Gym legged robots.
    Returns array of episode returns.
    """
    actor.eval()

    # Build environment
    env_cfg, _ = task_registry.get_cfgs(name=task_name)
    env_cfg.env.num_envs = num_envs
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # Optional: make it deterministic
    env_cfg.env.episode_length_s = 20  # or whatever you used in training
    env_cfg.commands.ranges.lin_vel_x = [0.5, 0.5]  # fix command
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    env, _ = task_registry.make_env(name=task_name, args=None, env_cfg=env_cfg)
    env.commands[:, :] = 0.0
    env.commands[:, 0] = 0.5  # forward velocity command

    # Move actor to correct device and set to eval
    actor.to(device)
    actor.eval()

    episode_rewards = []
    episode_lengths = []

    obs = env.get_observations()
    done = torch.zeros(num_envs, device=env.device, dtype=torch.bool)

    for ep in range(n_episodes):
        obs = env.reset()
        ep_reward = 0.0
        ep_len = 0

        while not done.all():
            with torch.no_grad():
                actions = actor.act_inference(obs)  # <-- critical: use your actor's inference
                if isinstance(actions, tuple):
                    actions = actions[0]

            obs, _, rews, dones, infos = env.step(actions)

            ep_reward += rews.sum().item() / num_envs
            ep_len += 1
            done = dones

            if done.any():
                done_indices = dones.nonzero(as_tuple=False).flatten()
                for idx in done_indices:
                    if episode_lengths and ep_len <= episode_lengths[-1]:
                        continue
                    episode_rewards.append(ep_reward / (ep_len + 1e-8))
                    episode_lengths.append(ep_len)
                obs = env.reset_done(done_indices)
                done = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
                ep_reward = 0.0
                ep_len = 0

        # In case episode ends naturally
        if ep_len > 0:
            episode_rewards.append(ep_reward / ep_len)
            episode_lengths.append(ep_len)

    actor.train()  # switch back
    return np.array(episode_rewards)
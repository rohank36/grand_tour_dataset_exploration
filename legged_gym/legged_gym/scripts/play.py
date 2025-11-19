import argparse
import sys

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym

from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger, export_policy_as_onnx
import numpy as np
import torch


def get_custom_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_policy', default=True)
    parser.add_argument('--export_onnx', default=True)
    parser.add_argument('--move_camera', action='store_true')
    parser.add_argument('--record_frames', action='store_true')
    parser.add_argument('--num_repetitions', type=int, default=1)

    # Parse only known arguments and leave the rest
    custom_args, remaining_args = parser.parse_known_args()
    # Update sys.argv to contain only the remaining arguments
    sys.argv = [sys.argv[0]] + remaining_args
    return custom_args


def create_video(recording_dir):
    import cv2
    import glob

    img_array = []
    for filename in sorted(glob.glob(os.path.join(recording_dir, '*.png'))):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
        os.remove(filename)

    out = cv2.VideoWriter(os.path.join(recording_dir, 'video.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def play(custom_args, args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    model_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, args.load_run)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    if args.domain_rand_friction_range is not None:
        env_cfg.domain_rand.randomize_friction = True
        env_cfg.domain_rand.randomize_added_mass = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_ground_friction = False
        env_cfg.domain_rand.friction_range = args.domain_rand_friction_range
    elif args.domain_rand_added_mass_range is not None:
        env_cfg.domain_rand.randomize_added_mass = True
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.randomize_ground_friction = False
        env_cfg.domain_rand.added_mass_range = args.domain_rand_added_mass_range
    elif args.domain_rand_push_range is not None:
        env_cfg.domain_rand.push_robots = True
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_added_mass = False
        env_cfg.domain_rand.randomize_ground_friction = False
        # env_cfg.domain_rand.push_interval_s = 10
        env_cfg.domain_rand.max_push_vel_xy = args.domain_rand_push_range
    elif args.domain_rand_ground_friction_range is not None:
        env_cfg.domain_rand.randomize_ground_friction = True
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_added_mass = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.ground_friction_range = args.domain_rand_ground_friction_range

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, play=True)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # export policy as a jit module (used to run it from C++)
    export_path = os.path.join(model_dir, 'exported', 'policies')
    if custom_args.export_policy:
        export_policy_as_jit(ppo_runner.alg.actor_critic, export_path)
        print('Exported policy as jit script to: ', export_path)
    if custom_args.export_onnx:
        # reuse the same path or choose a subfolder
        # grab one batch of obs (already on CPU)
        example_obs = obs.cpu()[0]
        export_policy_as_onnx(ppo_runner.alg.actor_critic, export_path, example_obs)

    logger = Logger(env.dt)
    robot_index = 0  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    recording_dir = os.path.join(model_dir, 'exported', 'recording')
    if custom_args.record_frames:
        os.makedirs(recording_dir, exist_ok=True)

    num_envs = env_cfg.env.num_envs
    max_episode_length = int(env.max_episode_length)

    episode_lengths = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    for i in range(custom_args.num_repetitions * int(max_episode_length)):
        actions = policy.act_inference(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        episode_lengths = torch.clamp(episode_lengths + 1, max=max_episode_length)
        done_indices = torch.where(dones.squeeze() == 1)[0]
        for env_idx in done_indices:
            env_idx = env_idx.item()
            episode_lengths[env_idx] = 0

        if custom_args.record_frames:
            if i % 2:
                filename = os.path.join(recording_dir, f'{img_idx:04d}.png')
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1
        if custom_args.move_camera:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i == stop_state_log:
            logger.plot_states(model_dir)
        if 0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i == stop_rew_log:
            logger.print_rewards()

    if custom_args.record_frames:
        create_video(recording_dir)

    torch.cuda.synchronize()  # Ensure all CUDA operations are completed




if __name__ == '__main__':
    custom_args = get_custom_args()
    args = get_args()
    play(custom_args, args)

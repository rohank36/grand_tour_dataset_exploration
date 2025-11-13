import zarr
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

#------------ reward functions----------------
class rew_functions:
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return np.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return np.sum(np.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return np.sum(np.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = np.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return np.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        # Penalize torques
        return np.sum(np.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return np.sum(np.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return np.sum(np.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return np.sum(np.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return np.sum(1.*(np.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return np.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return np.sum((np.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return np.sum((np.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = np.sum(np.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return np.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = np.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return np.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = np.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = np.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= np.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return np.any(np.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
                5 *np.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return np.sum(np.abs(self.dof_pos - self.default_dof_pos), dim=1) * (np.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return np.sum((np.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_hip_abduction_adduction(self):
        # Penalize wide spread between hip abduction/adduction joints
        hip_abduction_indices = [0, 3, 6, 9]  # Indices for LF_HAA, LH_HAA, RF_HAA, RH_HAA
        hip_positions = self.dof_pos[:, hip_abduction_indices]

        # Calculate the spread between front legs and between hind legs
        spread_front = np.abs(hip_positions[:, 0] - hip_positions[:, 2])  # LF - RF
        spread_hind = np.abs(hip_positions[:, 1] - hip_positions[:, 3])  # LH - RH

        # Penalize spreads exceeding a threshold
        threshold = self.cfg.rewards.max_leg_spread
        front_penalty = np.clamp(spread_front - threshold, min=0)
        hind_penalty = np.clamp(spread_hind - threshold, min=0)
        return front_penalty + hind_penalty

    def _reward_contact_sequence(self):
        # Reward proper contact sequence for walking gait
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.

        # Define desired contact sequence (e.g., diagonal pairs for trot gait)
        desired_sequence = np.tensor([[1, 0, 0, 1],
                                            [0, 1, 1, 0]], device=self.device)
        desired_sequence = desired_sequence.unsqueeze(0).expand(contact.size(0), -1, -1)


        # Check if current contact matches any phase of the desired sequence
        sequence_match = np.any(np.all(contact.unsqueeze(1) == desired_sequence, dim=2), dim=1)

        # Penalize incorrect sequences and reward correct ones
        reward = np.where(sequence_match,
                                self.cfg.rewards.correct_sequence_reward,
                                -self.cfg.rewards.incorrect_sequence_penalty)
        return reward

    def _reward_foot_drag(self):
        # Penalize dragging motion of feet when in contact with the ground

        # Get contact information (similar to _reward_feet_air_time)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.

        # Get foot velocities (approximating using the knee joint velocities)
        knee_indices = [2, 5, 8, 11]  # KFE joints
        foot_velocities = self.dof_vel[:, knee_indices]

        # Penalize motion for feet in contact
        drag_penalty = np.where(contact, np.abs(foot_velocities), np.zeros_like(foot_velocities))

        # Sum penalties across all feet
        total_drag_penalty = np.sum(drag_penalty, dim=1)
        return total_drag_penalty

    def _reward_body_orientation(self):
        # Penalize tilt angle
        tilt_penalty = np.sum(np.square(self.projected_gravity[:, :2]), dim=1)
        return tilt_penalty


# from https://github.com/modanesh/legged_gym_anymal_d/blob/master/legged_gym/envs/base/legged_robot_config.py#L132
class rewards:
    class scales:
        termination = -0.0
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        lin_vel_z = -2.0
        ang_vel_xy = -0.05
        orientation = -0.
        torques = -0.00001
        dof_vel = -0.
        dof_acc = -2.5e-7
        base_height = -0. 
        feet_air_time =  1.0
        collision = -1.
        feet_stumble = -0.0 
        action_rate = -0.01
        stand_still = -0.

        hip_abduction_adduction = -0.0
        foot_drag = -0.0
        body_orientation = -0.0
    
    max_leg_spread = 0.5
    correct_sequence_reward = 0.5
    incorrect_sequence_penalty = -0.5
    only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
    tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
    soft_dof_vel_limit = 1.
    soft_torque_limit = 1.
    base_height_target = 1.
    max_contact_force = 100. # forces above this value are penalized

def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

reward_scales = class_to_dict(rewards.scales)
decimation = 4
physics_base = 0.005
dt = decimation * physics_base
print(dt)

def _prepare_reward_function():
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(reward_scales.keys()):
            scale = reward_scales[key]
            if scale==0:
                reward_scales.pop(key) 
            else:
                reward_scales[key] *= dt
        # prepare list of functions
        reward_functions = []
        reward_names = []
        for name, scale in reward_scales.items():
            if name=="termination":
                continue
            reward_names.append(name)
            name = '_reward_' + name
            reward_functions.append(getattr(rew_functions, name))

        # reward episode sums
        """
        episode_sums = {
            name: np.zeros(num_envs, dtype=np.float32)
            for name in reward_scales.keys()
        }
        """
        return reward_names,reward_functions

def compute_reward(only_positive_rewards=True):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        reward_names, reward_functions = _prepare_reward_function()
        rew_buf[:] = 0.
        for i in range(len(reward_functions)):
            name = reward_names[i]
            rew = reward_functions[i]() * reward_scales[name]
            rew_buf += rew
            #episode_sums[name] += rew
        if only_positive_rewards:
            rew_buf[:] = np.clip(rew_buf[:], min=0.)
        # add termination reward after clipping
        #if "termination" in reward_scales:
            #rew = _reward_termination() * reward_scales["termination"]
            #rew_buf += rew
            #episode_sums["termination"] += rew



import os
import sys
import hydra
import logging
from pathlib import Path
from omegaconf import DictConfig
import time
import numpy as np
import gym
import imageio

from rolf.utils import make_env
from lib.dynsys.gmm_actor import GMMSkillActor

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class TaskEval(object):
    """
    Python wrapper that allows you to train DS skills on a given dataset
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.env = make_env(cfg.env.id, cfg.env, cfg.seed, cfg.rolf.name)
        self.env_ac_space = self.env.action_space
        self.skills = self.cfg.rolf.pretrain.skills
        self.env.target_tasks = np.copy(self.skills)
        self.env.tasks_to_complete = np.copy(self.skills)
        self.task_name = "_".join(self.skills)
        self.actor = GMMSkillActor(self.cfg.rolf.pretrain)
        self.actor.load_params()
        self.state_type = self.cfg.rolf.pretrain.dataset.input_type
        self.logs_out_dir = os.path.join(
            "log/task-eval", time.strftime("%m-%d-%H%M%S", time.gmtime(time.time()))
        )
        os.makedirs(self.logs_out_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.logs_out_dir, "logs.log"))
        fh.setLevel(logging.DEBUG)
        self.logger = logging.getLogger("TaskEval")
        self.logger.addHandler(fh)
        self.frames = []
        self.counter = 0
        self.ori_dt = 0.05

    def act(self, obs, skill_id):
        if self.state_type == "pos":
            x = obs["ob"][:3]
        dx, is_goal_reached = self.actor.act(x, skill_id)

        intermediate_angles = self.interpolate_euler_angles(
            obs["ob"][3:6], self.actor.skill_ds[skill_id].fixed_ori, 10
        )
        self.logger.info(f"Intermediate Goal: {intermediate_angles[1]}")
        goal = intermediate_angles[1]
        # goal = self.actor.skill_ds[skill_id].fixed_ori
        dx_ori = self.get_minimal_rotation(obs["ob"][3:6], goal)

        env_ac = np.append(
            dx, np.append(dx_ori, -1)
        )  # pos, ori, gripper_width i.e., size=7
        env_ac = gym.spaces.unflatten(self.env_ac_space, env_ac)
        return env_ac, is_goal_reached

    def run(self):
        succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
        rollout_return = 0
        skill_id = 0
        num_evals = 10
        for eval_no in range(num_evals):
            obs = self.calibrate_env_start_state()
            skill_id = 0
            for step in range(self.env.max_episode_steps):
                if self.cfg.rolf.record:
                    self.record_frame()
                if self.cfg.rolf.render:
                    self.env.render()

                action, is_goal_reached = self.act(obs, skill_id)
                self.logger.info(
                    f"{eval_no} {step} {obs['ob'][3:6]} {action['ac'][3:6] * self.ori_dt} {self.actor.skill_ds[skill_id].fixed_ori}"
                )
                obs, reward, done, info = self.env.step(action)
                rollout_return += reward
                print(
                    eval_no,
                    step,
                    np.linalg.norm(obs["ob"][:3] - self.actor.skill_ds[skill_id].goal),
                    is_goal_reached,
                )

                if is_goal_reached:
                    skill_id += 1
                    if skill_id == 4:
                        done = True

                if done:
                    break
            status = None
            if info["success"]:
                succesful_rollouts += 1
                status = "Success"
            else:
                status = "Fail"
            self.logger.info(
                f"{eval_no}: Skills Executed: {skill_id+1} Status: {status} Reward: {rollout_return}!"
            )

            if self.cfg.rolf.record:
                self.save_recording(eval_no)
                self.frames = []

        rollout_returns.append(rollout_return)
        rollout_lengths.append(step)
        acc = succesful_rollouts / num_evals
        self.logger.info(f"Accuracy: {acc * 100}")
        self.logger.info(f"Avg. Return: {np.mean(rollout_returns)}")
        self.logger.info(f"Avg. Steps: {np.mean(rollout_lengths)}")
        self.logger.info(f"Task: {self.task_name}")
        self.logger.shutdown()

    def calibrate_env_start_state(self, error_margin=0.01, max_checks=15):
        obs = self.env.reset()
        desired_start = self.actor.skill_ds[0].dataset.X[self.counter, 0, :]
        self.counter += 1
        dx_pos = desired_start - obs["ob"][:3]
        intermediate_angles = self.interpolate_euler_angles(
            obs["ob"][3:6], self.actor.skill_ds[0].fixed_ori, 5
        )
        goal = intermediate_angles[1]
        # goal = self.actor.skill_ds[0].fixed_ori
        dx_ori = self.get_minimal_rotation(obs["ob"][3:6], goal)

        action = np.append(
            dx_pos, np.append(dx_ori, -1)
        )  # pos, ori, gripper_width i.e., size=7
        action = gym.spaces.unflatten(self.env_ac_space, action)

        check_counter = 0
        while np.linalg.norm(obs["ob"][:3] - desired_start) > error_margin:
            obs, _, _, _ = self.env.step(action)
            dx_pos = desired_start - obs["ob"][:3]
            intermediate_angles = self.interpolate_euler_angles(
                obs["ob"][3:6], self.actor.skill_ds[0].fixed_ori, 5
            )
            goal = intermediate_angles[1]
            # goal = self.actor.skill_ds[0].fixed_ori
            dx_ori = self.get_minimal_rotation(obs["ob"][3:6], goal)

            action = np.append(
                dx_pos, np.append(dx_ori, -1)
            )  # pos, ori, gripper_width i.e., size=7
            action = gym.spaces.unflatten(self.env_ac_space, action)
            check_counter += 1
            if check_counter >= max_checks:
                # self.cons_logger.info(
                #     f"CALVIN is struggling to place the EE at the right initial pose. \
                #         Difference: {np.linalg.norm(obs - desired_start)}"
                # )
                break
        return obs

    def record_frame(self, obs_type="rgb", cam_type="static", size=200):
        """Record RGB obsservations"""
        rgb_obs, depth_obs = self.env.get_camera_obs()
        if obs_type == "rgb":
            frame = rgb_obs[f"{obs_type}_{cam_type}"]
        else:
            frame = depth_obs[f"{obs_type}_{cam_type}"]
        self.frames.append(frame)

    def save_recording(self, fname):
        """Save recorded frames as a video"""
        if len(self.frames) == 0:
            # This shouldn't happen but if it does, the function
            # call exits gracefully
            return None
        fname = f"{fname}.gif"
        kargs = {"duration": 33}
        fpath = os.path.join(self.logs_out_dir, fname)
        imageio.mimsave(fpath, np.array(self.frames), "GIF", **kargs)
        return fpath

    # TODO: Fix the problem with the euler angles
    def normalize_angle(self, angle):
        """Normalizes the input angle to the range [-pi, pi)."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_minimal_rotation(self, current_angles, goal_angles):
        """Computes the minimal rotation required to get from current_angles to goal_angles."""
        return self.normalize_angle(goal_angles - current_angles) / self.ori_dt

    def interpolate_euler_angles(self, current_euler, goal_euler, time_steps=5):
        # Convert Euler angles to quaternions
        q1 = R.from_euler("xyz", current_euler).as_quat()
        q2 = R.from_euler("xyz", goal_euler).as_quat()

        # Convert the quaternions to Rotation objects and combine them
        rotations = R.from_quat([q1, q2])

        # Initialize Slerp with rotations and associated times
        key_times = [0, 1]
        slerp = Slerp(key_times, rotations)

        # Compute interpolated rotations for the given times
        times = np.linspace(0, 1, time_steps)
        interpolated_rotations = slerp(times)

        # Convert rotations back to eulers and return
        return interpolated_rotations.as_euler("xyz")


@hydra.main(config_path="config", config_name="seqref_calvin")
def main(cfg: DictConfig) -> None:
    eval = TaskEval(cfg)
    eval.run()


if __name__ == "__main__":
    main()

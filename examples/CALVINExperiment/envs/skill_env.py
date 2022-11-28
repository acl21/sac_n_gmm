from gym import spaces
from calvin_env.envs.play_table_env import PlayTableSimEnv
import hydra
import imageio
import os
import cv2
import numpy as np


class SkillSpecificEnv(PlayTableSimEnv):
    def __init__(self, tasks: dict = {}, **kwargs):
        super(SkillSpecificEnv, self).__init__(**kwargs)
        # For this example we will modify the observation to
        # only retrieve the end effector pose
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,))
        # We can use the task utility to know if the task was executed correctly
        self.tasks = hydra.utils.instantiate(tasks)
        self.skill_name = None
        self.state_type = None
        self.max_episode_steps = 1000

        self.frames = []
        self.outdir = None
        self.record_count = 1


    def set_skill(self, skill):
        """Set skill name"""
        self.skill_name = skill

    def set_state_type(self, type):
        """Set env input type - joint, pos, pos_ori"""
        self.state_type = type
        self.obs_idx = self.get_valid_columns()

    def reset(self):
        obs = super().reset()
        self.start_info = self.get_info()
        return obs

    def get_obs(self):
        """Overwrite robot obs to only retrieve end effector position"""
        robot_obs, robot_info = self.robot.get_observation()
        return robot_obs

    def get_camera_obs(self):
        """Collect camera, robot and scene observations."""
        assert self.cameras is not None
        rgb_obs = {}
        depth_obs = {}
        for cam in self.cameras:
            rgb, depth = cam.render()
            rgb_obs[f"rgb_{cam.name}"] = rgb
            depth_obs[f"depth_{cam.name}"] = depth
        obs = {"rgb_obs": rgb_obs, "depth_obs": depth_obs}
        return obs

    def set_outdir(self, outdir):
        """Set output directory where recordings can/will be saved"""
        self.outdir = outdir

    def record_frame(self, obs_type='rgb', cam_type='static', size=200):
        """Record RGB obsservation"""
        frame = self.get_camera_obs()[f'{obs_type}_obs'][f'{obs_type}_{cam_type}']
        frame = cv2.resize(frame, (size, size), interpolation = cv2.INTER_AREA)
        self.frames.append(frame)

    def reset_recorded_frames(self):
        """Reset recorded frames"""
        self.frames = []

    def save_recorded_frames(self, path=None):
        """Save recorded frames as a video"""
        if path is None:
            imageio.mimsave(os.path.join(self.outdir, f'{self.skill_name}_{self.state_type}_{self.record_count}.mp4'), self.frames, fps=30)
            self.record_count += 1
        else:
            imageio.mimsave(path, self.frames, fps=30)
        return os.path.join(self.outdir, f'{self.skill_name}_{self.state_type}_{self.record_count-1}.mp4')

    def _success(self):
        """Returns a boolean indicating if the task was performed correctly"""
        current_info = self.get_info()
        task_filter = [self.skill_name]
        task_info = self.tasks.get_task_info_for_set(self.start_info, current_info, task_filter)
        return self.skill_name in task_info

    def _reward(self):
        """Returns the reward function that will be used
        for the RL algorithm"""
        reward = int(self._success()) * 10
        r_info = {"reward": reward}
        return reward, r_info

    def _termination(self):
        """Indicates if the robot has reached a terminal state"""
        success = self._success()
        done = success
        d_info = {"success": success}
        return done, d_info

    def get_valid_columns(self):
        if 'joint' in self.state_type:
            start, end = 8, 15
        elif 'pos_ori' in self.state_type:
            start, end = 1, 7
        elif 'pos' in self.state_type:
            start, end = 1, 4
        elif 'ori' in self.state_type:
            start, end = 4, 7
        elif 'grip' in self.state_type:
            start, end = 7, 8
        return start-1, end-1

    def prepare_action(self, input, type):
        if self.state_type == 'joint':
            action = {'type': f'joint_{type}', 'action': None}
            action['action'] = np.append(input, -1)
        elif self.state_type == 'pos':
            action = {'type': f'cartesian_{type}', 'action': None}
            action['action'] = np.append(input, np.append(np.zeros(3), -1))
        elif self.state_type == 'ori':
            action = {'type': f'cartesian_{type}', 'action': None}
            action['action'] = np.append(np.zeros(3), np.append(input, -1))
        elif self.state_type == 'pos_ori':
            action = {'type': f'cartesian_{type}', 'action': None}
            action['action'] = np.append(input, -1)

        return action


    def step(self, action):
        """Performing a relative action in the environment
        input:
            action: 7 tuple containing
                    Position x, y, z.
                    Angle in rad x, y, z.
                    Gripper action
                    each value in range (-1, 1)
        output:
            observation, reward, done info
        """
        # Transform gripper action to discrete space
        env_action = action.copy()
        env_action['action'][-1] = (int(action['action'][-1] >= 0) * 2) - 1
        self.robot.apply_action(env_action)
        for i in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        obs = self.get_obs()
        info = self.get_info()
        reward, r_info = self._reward()
        done, d_info = self._termination()
        info.update(r_info)
        info.update(d_info)
        return obs, reward, done, info
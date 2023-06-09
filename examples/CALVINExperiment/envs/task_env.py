from gym import spaces
import hydra
import copy
import imageio
import cv2
import os
import numpy as np

from examples.CALVINExperiment.calvin_env.calvin_env.envs.play_table_env import (
    PlayTableSimEnv,
)


class TaskSpecificEnv(PlayTableSimEnv):
    def __init__(self, tasks={}, target_tasks=[], sequential=True, **kwargs):
        super(TaskSpecificEnv, self).__init__(**kwargs)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(15,))
        self.tasks = hydra.utils.instantiate(tasks)
        self.target_tasks = target_tasks
        self.tasks_to_complete = copy.deepcopy(self.target_tasks)
        self.completed_tasks_so_far = []
        self.sequential = sequential
        self.state_type = "pos"
        self.frames = []
        self._max_episode_steps = 0
        self.subgoal_reached = False

    def _success(self):
        """
        Returns a boolean indicating if the task was performed correctly.
        Part of this function's logic was taken from https://github.com/clvrai/skimo/blob/main/envs/calvin.py#L61.
        """
        current_info = self.get_info()
        completed_tasks = self.tasks.get_task_info_for_set(
            self.start_info, current_info, self.tasks_to_complete
        )
        next_task = self.tasks_to_complete[0]
        for task in list(completed_tasks):
            if self.sequential:
                if task == next_task:
                    self.tasks_to_complete.pop(0)
                    self.completed_tasks_so_far.append(task)
            else:
                if task in self.tasks_to_complete:
                    self.tasks_to_complete.remove(task)
                    self.completed_tasks_so_far.append(task)
            if len(self.completed_tasks_so_far) == 1:
                if task == "open_drawer":
                    self.start_info["scene_info"]["doors"]["base__drawer"][
                        "current_state"
                    ] = 0.2
                elif task == "close_drawer":
                    self.start_info["scene_info"]["doors"]["base__drawer"][
                        "current_state"
                    ] = 0.0
            if len(self.tasks_to_complete) == 0:
                return True
            else:
                next_task = self.tasks_to_complete[0]
                self.subgoal_reached = True

        return len(self.tasks_to_complete) == 0

    def _reward(self):
        """Returns the reward function that will be used
        for the RL algorithm"""
        reward = int(self._success())
        # if self.subgoal_reached:
        #     reward += 1
        #     self.subgoal_reached = False
        r_info = {"reward": reward}
        return reward, r_info

    def _termination(self):
        """Indicates if the robot has reached a terminal state"""
        done = len(self.tasks_to_complete) == 0
        d_info = {
            "success": done,
            "tasks_to_complete": self.tasks_to_complete,
            "completed_tasks": self.completed_tasks_so_far,
        }
        return done, d_info

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
        env_action["action"][-1] = (int(action["action"][-1] >= 0) * 2) - 1
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

    def prepare_action(self, input, type):
        action = []
        if self.state_type == "joint":
            action = {"type": f"joint_{type}", "action": None}
        elif "pos" in self.state_type:
            action = {"type": f"cartesian_{type}", "action": None}
            action["action"] = input

        return action

    def get_obs(self):
        """Overwrite robot obs to only retrieve end effector position"""
        robot_obs, robot_info = self.robot.get_observation()
        if self.state_type == "pos":
            return robot_obs[:3]
        else:
            return robot_obs

    def reset(self):
        obs = super().reset()
        self.start_info = self.get_info()
        self.tasks_to_complete = copy.deepcopy(self.target_tasks)
        self.completed_tasks_so_far = []
        self.reset_recorded_frames()
        return obs

    def get_camera_obs(self):
        """Collect camera, robot and scene observations."""
        assert self.cameras is not None
        rgb_obs = {}
        depth_obs = {}
        for cam in self.cameras:
            rgb, depth = cam.render()
            rgb_obs[f"rgb_{cam.name}"] = rgb
            depth_obs[f"depth_{cam.name}"] = depth
        return rgb_obs, depth_obs

    def record_frame(self, obs_type="rgb", cam_type="static", size=200):
        """Record RGB obsservation"""
        rgb_obs, depth_obs = self.get_camera_obs()
        if obs_type == "rgb":
            frame = rgb_obs[f"{obs_type}_{cam_type}"]
        else:
            frame = depth_obs[f"{obs_type}_{cam_type}"]
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        self.frames.append(frame)

    def reset_recorded_frames(self):
        """Reset recorded frames"""
        self.frames = []

    def save_recorded_frames(self, outdir, fname):
        """Save recorded frames as a video"""
        if len(self.frames) == 0:
            # This shouldn't happen but if it does, the function
            # call exits gracefully
            return None
        fname = f"{fname}.gif"
        kargs = {"fps": 30}
        fpath = os.path.join(outdir, fname)
        imageio.mimsave(fpath, np.array(self.frames), "GIF", **kargs)
        return fpath

import gym
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
    def __init__(self, tasks, target_tasks, sequential, sparse_rewards, **kwargs):
        super(TaskSpecificEnv, self).__init__(**kwargs)
        self.action_space = self.get_action_space()
        self.observation_space = self.get_obs_space()
        self.tasks = hydra.utils.instantiate(tasks)
        self.target_tasks = target_tasks
        self.tasks_to_complete = copy.deepcopy(self.target_tasks)
        self.sparse_rewards = sparse_rewards
        self.completed_tasks = []
        self.sequential = sequential
        self.state_type = "pos"
        self.frames = []
        self._max_episode_steps = 0
        self.subgoal_reached = False

    def get_action_space(self):
        """
        Returns env's action space as a gym.spaces.Box object
        """
        return gym.spaces.Box(low=-1, high=1, shape=(7,))

    def get_obs_space(self):
        """
        Returns env's observation space as a gym.spaces.Box object
        """
        return gym.spaces.Box(low=0, high=1, shape=self.get_obs().shape)

    def get_camera_obs(self):
        """Collect camera, robot and scene observations.

        Camera Observations
        rgb_static: (dtype=np.uint8, shape=(200, 200, 3)),
        rgb_gripper: (dtype=np.uint8, shape=(84, 84, 3)),
        rgb_tactile: (dtype=np.uint8, shape=(160, 120, 6)),
        depth_static: (dtype=np.float32, shape=(200, 200)),
        depth_gripper: (dtype=np.float32, shape=(84, 84)),
        depth_tactile: (dtype=np.float32, shape=(160, 120, 2))
        """
        assert self.cameras is not None
        rgb_obs = {}
        depth_obs = {}
        for cam in self.cameras:
            rgb, depth = cam.render()
            rgb_obs[f"rgb_{cam.name}"] = rgb
            depth_obs[f"depth_{cam.name}"] = depth
        return rgb_obs, depth_obs

    def get_obs(self):
        """
        Robot Observations ('robot_obs'):
        (dtype=np.float32, shape=(15,))
        tcp position (3): x,y,z in world coordinates
        tcp orientation (3): euler angles x,y,z in world coordinates
        gripper opening width (1): in meter
        arm_joint_states (7): in rad
        gripper_action (1): binary (close = -1, open = 1)

        Scene Observations ('scene_obs'):
        (dtype=np.float32, shape=(24,))
        sliding door (1): joint state
        drawer (1): joint state
        button (1): joint state
        switch (1): joint state
        lightbulb (1): on=1, off=0
        green light (1): on=1, off=0
        red block (6): (x, y, z, euler_x, euler_y, euler_z)
        blue block (6): (x, y, z, euler_x, euler_y, euler_z)
        pink block (6): (x, y, z, euler_x, euler_y, euler_z)
        """

        obs = self.get_state_obs()
        # Return only pos (size 3)
        # return obs['robot_obs'][:3]
        # Return pos + joint states (size 10)
        # return np.concatenate([obs["robot_obs"][:3], obs["robot_obs"][7:-1]])
        # Return pos + joint + drawer state (size 11)
        ob = np.concatenate(
            [obs["robot_obs"][:3], obs["robot_obs"][7:-1], obs["scene_obs"][1:2]]
        )
        ob[-1] = int(ob[-1] >= 0.12)  # 1 when open, 0 otherwise
        return ob

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
        self.scene.step()
        obs = self.get_obs()
        info = self.get_info()
        reward, r_info = self._reward()
        done, d_info = self._termination()
        info.update(r_info)
        info.update(d_info)
        return obs, reward, done, info

    def reset(self):
        obs = super().reset()
        self.scene.reset()
        self.start_info = self.get_info()
        self.tasks_to_complete = copy.deepcopy(self.target_tasks)
        self.completed_tasks = []
        self.reset_recording()
        return obs

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
                    self.completed_tasks.append(task)
            else:
                if task in self.tasks_to_complete:
                    self.tasks_to_complete.remove(task)
                    self.completed_tasks.append(task)
            if len(self.completed_tasks) == 1:
                # This manual inteference helps with tasks when open_drawer
                # and close_drawer skills are executed one after another
                if task == "open_drawer":
                    self.start_info["scene_info"]["doors"]["base__drawer"][
                        "current_state"
                    ] = 0.18
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
        """
        Returns the reward function that will be used
        for the RL algorithm
        """
        reward = int(self._success())
        if self.subgoal_reached and not self.sparse_rewards:
            reward += 1
            self.subgoal_reached = False
        r_info = {"reward": reward}
        return reward, r_info

    def _termination(self):
        """
        Indicates if the robot has reached a terminal state
        """
        done = len(self.tasks_to_complete) == 0
        d_info = {
            "success": done,
            "tasks_to_complete": self.tasks_to_complete,
            "completed_tasks": self.completed_tasks,
        }
        return done, d_info

    def prepare_action(self, input, action_type="abs", state_type=None):
        """
        Returns a dict in a format expected by the CALVIN env.
        """
        if state_type is None:
            state_type = self.state_type
        assert state_type in [
            "pos",
            "pos_ori",
            "cartesian",
            "joint",
        ], "Invalid state type!"
        assert action_type in ["abs", "rel"], "Invalid action type!"
        if "pos" in state_type:
            state_type = "cartesian"
        action = []
        action = {"type": f"{state_type}_{action_type}", "action": None}
        action["action"] = input

        return action

    def record_frame(self, obs_type="rgb", cam_type="static", size=200):
        """Record RGB obsservations"""
        rgb_obs, depth_obs = self.get_camera_obs()
        if obs_type == "rgb":
            frame = rgb_obs[f"{obs_type}_{cam_type}"]
        else:
            frame = depth_obs[f"{obs_type}_{cam_type}"]
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        self.frames.append(frame)

    def save_recording(self, outdir, fname):
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

    def reset_recording(self):
        """Reset recorded frames"""
        self.frames = []

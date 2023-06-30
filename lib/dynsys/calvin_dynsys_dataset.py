import os
import glob
import torch
import pybullet
import numpy as np


class CALVINDynSysDataset(object):
    def __init__(
        self,
        skill,
        train=True,
        state_type="joint",
        demos_dir="",
        goal_centered=False,
        dt=0.02,
        normalized=False,
        is_quaternion=False,
        ignore_bad_demos=False,
    ):
        self.skill = skill
        self.state_type = state_type
        self.demos_dir = demos_dir
        self.goal_centered = goal_centered
        self.dt = dt
        self.normalized = normalized
        self.train = train
        self.ignore_bad_demos = ignore_bad_demos

        # Check if demos and skill directory exist
        assert os.path.isdir(self.demos_dir), "Demos directory does not exist!"
        assert os.path.isdir(
            os.path.join(self.demos_dir, self.skill)
        ), "Skill directory does not exist!"

        # Load from the demo directory
        if self.train:
            fname = "training"
        else:
            fname = "validation"
        self.data_file = glob.glob(
            os.path.join(self.demos_dir, self.skill, f"{fname}.npy")
        )[0]
        start_idx, end_idx = self.get_valid_columns(self.state_type)
        self.X = np.load(self.data_file)[:, :, start_idx:end_idx]

        # Ignore trajectories that do not succeed in the env (only for train dataset)
        if self.ignore_bad_demos and self.train:
            ignore_indices = []
            ignore_indices = self.get_bad_demo_indices()
            self.X = np.delete(self.X, ignore_indices, axis=0)

        # Get the euler angles best for the skill
        if self.skill in ["open_drawer", "close_drawer", "turn_on_led"]:
            self.fixed_ori = np.array([3.142, 0.08, 1.5])
        elif self.skill in ["turn_on_lightbulb", "move_slider_left"]:
            self.fixed_ori = np.array([3.0, -0.4, 1.5])

        # Convert fixed ori from Euler to Quaternion when flagged
        # Convert Euler orientations in self.X to Quaternion when flagged
        if is_quaternion:
            self.fixed_ori = np.array(pybullet.getQuaternionFromEuler(self.fixed_ori))
            if self.state_type == "ori":
                self.X = np.apply_along_axis(
                    pybullet.getQuaternionFromEuler, -1, self.X
                )
            elif self.state_type == "pos_ori":
                oris = np.apply_along_axis(
                    pybullet.getQuaternionFromEuler, -1, self.X[:, :, 3:]
                )
                self.X = np.concatenate([self.X[:, :, :3], oris], axis=-1)

        # Average end point i.e., goal of the trajectory (useful during inference)
        if "joint" not in self.state_type:
            self.goal = np.mean(self.X[:, -1, :], axis=0)
        else:
            self.goal = 0

        if self.goal_centered:
            assert "joint" not in self.state_type, "Do not goal center joint data!"
            # Make X goal centered i.e., subtract each trajectory with its goal
            # Experiments show that goal centering is better with each trajectory's
            # own goal and not with the average self.goal. However, during inference
            # we use the self.goal
            self.X = self.X - np.expand_dims(self.X[:, -1, :], axis=1)

        if self.normalized:
            assert "joint" not in self.state_type, "Do not normalize joint data!"
            self.X_mins = np.min(self.X.reshape(-1, self.X.shape[-1]), axis=0)
            self.X_maxs = np.max(self.X.reshape(-1, self.X.shape[-1]), axis=0)
            self.norm_range = [-1, 1]
            self.X = self.normalize(self.X)

        # Get first order derivative dX from X
        self.dX = np.empty((self.X.shape[0], self.X.shape[1], self.X.shape[2]))
        self.dX[:, :-2, :] = (self.X[:, 2:, :] - self.X[:, :-2, :]) / self.dt
        self.dX[:, -2, :] = (self.X[:, -1, :] - self.X[:, -2, :]) / self.dt
        self.dX[:, -1, :] = np.zeros(self.dX.shape[-1])

    def normalize(self, x):
        """See this link for clarity: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1"""
        assert self.X_mins is not None, "Cannot normalize with X_mins as None"
        assert self.X_maxs is not None, "Cannot normalize with X_maxs as None"
        return (self.norm_range[-1] - self.norm_range[0]) * (x - self.X_mins) / (
            self.X_maxs - self.X_mins
        ) + self.norm_range[0]

    def undo_normalize(self, x):
        """See this link for clarity: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1"""
        assert self.X_mins is not None, "Cannot undo normalization with X_mins as None"
        assert self.X_maxs is not None, "Cannot undo normalization with X_maxs as None"
        return (x - self.norm_range[0]) * (self.X_maxs - self.X_mins) / (
            self.norm_range[-1] - self.norm_range[0]
        ) + self.X_mins

    def get_valid_columns(self, state_type):
        if "joint" in state_type:
            start, end = 8, 15
        elif "pos_ori" in state_type:
            start, end = 1, 7
        elif "pos" in state_type:
            start, end = 1, 4
        elif "ori" in state_type:
            start, end = 4, 7
        elif "grip" in state_type:
            start, end = 7, 8
        return start, end

    def get_bad_demo_indices(self):
        ignore_indices = []
        if self.skill == "open_drawer":
            ignore_indices = [3, 61, 72, 78, 85, 108, 129]
        elif self.skill == "turn_on_lightbulb":
            ignore_indices = [
                0,
                1,
                3,
                19,
                20,
                22,
                23,
                25,
                27,
                32,
                34,
                36,
                37,
                45,
                46,
                53,
                54,
                63,
                70,
                73,
                76,
                80,
                82,
                86,
                87,
                89,
                90,
                92,
                98,
                102,
                103,
                104,
                107,
                110,
                116,
                117,
                124,
                128,
            ]
        elif self.skill == "move_slider_left":
            ignore_indices = [
                2,
                5,
                9,
                10,
                12,
                13,
                17,
                18,
                20,
                21,
                22,
                24,
                35,
                36,
                37,
                38,
                39,
                43,
                44,
                46,
                49,
                51,
                54,
                56,
                61,
                62,
                69,
                70,
                71,
                72,
                73,
                75,
                78,
                79,
                80,
                83,
                87,
                89,
                90,
                93,
                94,
                97,
                98,
                99,
                100,
                101,
                107,
                109,
                111,
                113,
                114,
                119,
                122,
                124,
                128,
                130,
                133,
                142,
                144,
            ]
        elif self.skill == "turn_on_led":
            ignore_indices = [0, 3, 33, 74, 83, 87, 98]
        else:
            print("Must ignore trajectory that do not succeed in the env!")
        return ignore_indices

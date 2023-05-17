import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torch
import pdb
import pybullet as p


class CALVINDynSysDataset(Dataset):
    def __init__(
        self,
        skill,
        train=True,
        state_type="pos",
        demos_dir="/work/dlclarge1/lagandua-refine-skills/calvin_demos/",
        goal_centered=False,
        dt=2 / 30,
        sampling_dt=1 / 30,
        normalized=False,
        is_quaternion=False,
    ):
        self.skill = skill
        self.demos_dir = demos_dir
        self.goal_centered = goal_centered
        self.dt = dt
        self.sampling_dt = sampling_dt
        self.normalized = normalized
        self.norm_range = [-1, 1]
        self.X_mins = None
        self.X_maxs = None
        self.train = train
        self.fixed_ori = None
        self.start = None
        self.goal = None
        if self.train:
            fname = "training"
        else:
            fname = "validation"
        self.data_file = glob.glob(os.path.join(self.demos_dir, self.skill, f"{fname}.npy"))[0]
        self.state_type = state_type

        start_idx, end_idx = self.get_valid_columns(self.state_type)
        self.X = np.load(self.data_file)[:, :, start_idx:end_idx]

        # Get the last orientation from the trajectory (this is bad for orientation dependant tasks)
        s_idx, e_idx = self.get_valid_columns("ori")
        temp_ori = np.load(self.data_file)[:, :, s_idx:e_idx]
        self.fixed_ori = temp_ori[0, -1, :]

        if self.state_type == "ori" and is_quaternion:
            self.X = np.apply_along_axis(p.getQuaternionFromEuler, -1, self.X)
        elif self.state_type == "pos_ori" and is_quaternion:
            oris = np.apply_along_axis(p.getQuaternionFromEuler, -1, self.X[:, :, 3:])
            self.X = np.concatenate([self.X[:, :, :3], oris], axis=-1)

        self.start = np.mean(self.X[:, 0, :], axis=0)
        self.goal = np.mean(self.X[:, -1, :], axis=0)
        if self.goal_centered:
            # Make X goal centered i.e., subtract each trajectory with its goal
            self.X = self.X - np.expand_dims(self.X[:, -1, :], axis=1)

        if self.normalized:
            self.set_mins_and_maxs(self.X)
            self.X = self.normalize(self.X)

        self.dX = (self.X[:, 2:, :] - self.X[:, :-2, :]) / self.dt
        self.X = self.X[:, 1:-1, :]

        self.X = torch.from_numpy(self.X).type(torch.FloatTensor)
        self.dX = torch.from_numpy(self.dX).type(torch.FloatTensor)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.dX[idx]

    def set_mins_and_maxs(self, data=None):
        self.X_mins = np.min(data.reshape(-1, data.shape[-1]), axis=0)
        self.X_maxs = np.max(data.reshape(-1, data.shape[-1]), axis=0)

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

    def plot_random(self):
        sampled_path = []
        rand_idx = np.random.randint(0, len(self.X))
        true_x = self.X[rand_idx, :, :].numpy()
        x = true_x[0]
        for t in range(len(true_x)):
            sampled_path.append(x)
            delta_x = self.sampling_dt * self.dX[rand_idx, t, :].numpy()
            x = x + delta_x
        sampled_path = np.array(sampled_path)
        plot_3d_trajectories(true_x, sampled_path)


def plot_3d_trajectories(demos, repro=None, goal=None, figsize=(4, 4)):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    if goal is not None:
        ax.scatter(goal[0], goal[1], goal[2], marker="*", s=15, label="Goal", c="red")
    if repro is None:
        for i in range(demos.shape[0]):
            x_val = demos[i, :, 0]
            y_val = demos[i, :, 1]
            z_val = demos[i, :, 2]
            ax.scatter(x_val, y_val, z_val, s=10)
    else:
        ax.scatter(demos[:, 0], demos[:, 1], demos[:, 2], alpha=0.5, s=1, label="Demonstration")
        ax.scatter(repro[:, 0], repro[:, 1], repro[:, 2], alpha=0.5, s=1, label="Reproduction")
    plt.legend()
    plt.tight_layout()
    plt.show()

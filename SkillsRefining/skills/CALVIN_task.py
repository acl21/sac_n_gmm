import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torch


class CALVINTaskDataset(Dataset):
    def __init__(
        self,
        task,
        train=True,
        state_type="pos",
        demos_dir="",
        dt=2 / 30,
        sampling_dt=1 / 30,
    ):
        self.task = task
        self.task_name = "_".join(self.task)
        self.demos_dir = demos_dir
        self.dt = dt
        self.sampling_dt = sampling_dt
        self.state_type = state_type
        self.fixed_ori = np.array([3.068032, 0.03401114, 1.48432319])
        self.train = train
        if self.train:
            fname = "training"
        else:
            fname = "validation"
        self.data_file = glob.glob(
            os.path.join(self.demos_dir, self.task_name, f"{fname}.npy")
        )[0]
        self.X = np.load(self.data_file)
        self.dX = (self.X[:, 2:, :] - self.X[:, :-2, :]) / self.dt
        self.X = self.X[:, 1:-1, :]

        self.X = torch.from_numpy(self.X).type(torch.FloatTensor)
        self.dX = torch.from_numpy(self.dX).type(torch.FloatTensor)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.dX[idx]

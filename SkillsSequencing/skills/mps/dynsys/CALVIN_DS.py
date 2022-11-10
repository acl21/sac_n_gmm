import glob
import os
import numpy as np
from torch.utils.data import Dataset
import torch

from utils import plot_3d_trajectories

class CALVINDynSysDataset(Dataset):
    def __init__(self, skill, train=True, state_type='joint', demos_dir='/work/dlclarge1/lagandua-refine-skills/calvin_demos/'):
        self.skill = skill
        self.demos_dir = demos_dir
        if train:
            fname = 'training'
        else:
            fname = 'validation'
        data_file = glob.glob(os.path.join(self.demos_dir, self.skill, f'{fname}.npy'))[0]
        self.state_type = state_type

        dt = 2 / 30
        start_idx, end_idx = self.get_valid_columns()
        self.X = np.load(data_file)[:,:,start_idx:end_idx]
        # ipdb.set_trace()
        self.dX = (self.X[:, 2:, :] - self.X[:, :-2, :]) / dt
        self.X = self.X[:, 1:-1, :]
        
        self.X = torch.from_numpy(self.X).type(torch.FloatTensor)
        self.dX = torch.from_numpy(self.dX).type(torch.FloatTensor)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.dX[idx]

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
        return start, end

    def plot_random(self):
        sampled_path = []
        rand_idx = np.random.randint(0, len(self.X))
        true_x = self.X[rand_idx, :, :].numpy()
        x = true_x[0]
        sampling_dt = 1 / 30
        for t in range(len(true_x)):
            sampled_path.append(x)
            x_dot = self.dX[rand_idx, t, :].numpy()
            x = x + sampling_dt * x_dot
        sampled_path = np.array(sampled_path)
        plot_3d_trajectories(true_x, sampled_path)
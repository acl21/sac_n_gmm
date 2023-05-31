import numpy as np
import torch
import os


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = device

        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        if isinstance(action_shape, int):
            action_shape = (action_shape,)

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.save_dir = None
        self.last_saved_idx = self.idx

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        dones = torch.as_tensor(self.dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, dones

    def save(self):
        file = os.path.join(self.save_dir, f"{self.last_saved_idx}_{self.idx}.npz")
        np.savez(
            file=file,
            obs=self.obses[self.last_saved_idx : self.idx],
            actions=self.actions[self.last_saved_idx : self.idx],
            rewards=self.rewards[self.last_saved_idx : self.idx],
            next_obs=self.next_obses[self.last_saved_idx : self.idx],
            dones=self.dones[self.last_saved_idx : self.idx],
        )
        self.last_saved_idx = self.idx

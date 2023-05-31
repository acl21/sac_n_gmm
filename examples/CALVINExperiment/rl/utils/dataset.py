from examples.CALVINExperiment.rl.utils.replay_buffer import ReplayBuffer
from torch.utils.data.dataset import IterableDataset
from typing import Iterator


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training
    """

    def __init__(self, buffer: ReplayBuffer, batch_size: int = 64) -> None:
        """
        Args:
            buffer: replay buffer
            sample_size: number of experiences to sample at a time
        """
        self.buffer = buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterator:
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], next_states[i], dones[i]

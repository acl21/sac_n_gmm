import logging
from pathlib import Path
import numpy as np
from collections import deque, namedtuple
from tqdm import tqdm
import os

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    def __init__(self, max_capacity=5e6):
        self.logger = logging.getLogger(__name__)
        self.unsaved_transitions = 0
        self.curr_file_idx = 1
        self.replay_buffer = deque(maxlen=int(max_capacity))
        self.save_dir = None

    def __len__(self) -> int:
        return len(self.replay_buffer)

    def add(self, state, action, reward, next_state, done):
        """
        This method adds a transition to the replay buffer.
        """
        transition = Transition(state, action, reward, next_state, done)
        self.replay_buffer.append(transition)
        self.unsaved_transitions += 1

    def sample(self, batch_size: int):
        indices = np.random.choice(
            len(self.replay_buffer),
            min(len(self.replay_buffer), batch_size),
            replace=False,
        )
        states, actions, rewards, next_states, dones = zip(
            *[self.replay_buffer[idx] for idx in indices]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=bool),
        )

    def save(self):
        return None

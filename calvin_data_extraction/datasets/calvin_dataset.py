import logging
import os
import re
from pathlib import Path
import numpy as np
import torch
from typing import Dict, List, Tuple, Union
from datasets.base_dataset import BaseDataset
from datasets.utils.load_utils import load_npz

logger = logging.getLogger(__name__)

import pdb


class CalvinDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(CalvinDataset, self).__init__(*args, **kwargs)
        self.episode_lookup = self.load_file_indices(self.data_dir, self.skill)
        self.naming_pattern, self.n_digits = self.lookup_naming_pattern()

    def __len__(self):
        """
        returns
        ----------
        number of possible starting frames
        """
        self.num_demos = len(self.episode_lookup)

        return self.num_demos

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:

        return self.get_sequences(idx)

    def lookup_naming_pattern(self):
        it = os.scandir(self.data_dir)
        while True:
            filename = Path(next(it))
            if "npz" in filename.suffix:
                break
        aux_naming_pattern = re.split(r"\d+", filename.stem)
        naming_pattern = [filename.parent / aux_naming_pattern[0], filename.suffix]
        n_digits = len(re.findall(r"\d+", filename.stem)[0])
        assert len(naming_pattern) == 2
        assert n_digits > 0
        return naming_pattern, n_digits

    def get_episode_name(self, idx: int) -> Path:
        """
        Convert frame idx to file name
        """
        return Path(f"{self.naming_pattern[0]}{idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def zip_sequence(self, start_idx: int, end_idx: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive individual frames saved as npy files and combine to episode dict
        parameters:
        -----------
        start_idx: index of first frame
        end_idx: index of last frame
        returns:
        -----------
        episode: dict of numpy arrays containing the episode where keys are the names of modalities
        """
        episodes = [load_npz(self.get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx, self.step_len)]
        episode = {key: np.stack([ep[key] for ep in episodes]) for key, _ in episodes[0].items()}
        return episode

    def get_sequences(self, idx: int) -> Dict:
        """
        parameters
        ----------
        idx: index of starting frame
        returns
        ----------
        seq_state_obs:  numpy array of state observations
        seq_rgb_obs:    tuple of numpy arrays of rgb observations
        seq_depth_obs:  tuple of numpy arrays of depths observations
        seq_acts:       numpy array of actions
        """
        info_indx = self.episode_lookup[idx]
        start_file_indx = info_indx[0]
        end_file_indx = info_indx[1]

        episode = self.zip_sequence(start_file_indx, end_file_indx)
        robot_obs = [self.transform_robot_obs(obs) for obs in episode["robot_obs"][:, :7]]

        robot_obs = torch.stack(robot_obs)

        batch = {"robot_obs": robot_obs}
        return batch

    def load_file_indices(self, data_dir: Path, skill: str) -> Tuple[List, List]:
        """
        this method builds the mapping from index to file_name used for loading the episodes
        parameters
        ----------
        data_dir:               absolute path of the directory containing the datasets
        returns
        ----------
        episode_lookup:                 list for the mapping from training example index to episode (file) index
        max_batched_length_per_demo:    list of possible starting indices per episode
        """
        assert data_dir.is_dir()
        skill_name = skill

        episode_lookup = []

        file_name = data_dir / "lang_annotations" / "auto_lang_ann.npy"
        data = np.load(file_name, allow_pickle=True).reshape(-1)[0]

        all_eps_idx_part_task = [i for (i, v) in enumerate(data["language"]["task"]) if v == skill_name]
        all_eps_start_end_part_task = [data["info"]["indx"][i] for i in all_eps_idx_part_task]

        for i in range(len(all_eps_start_end_part_task)):
            if all_eps_start_end_part_task[i][1] - all_eps_start_end_part_task[i][0] == 64:
                episode_lookup.append(all_eps_start_end_part_task[i])

        logger.info(f"Found {len(episode_lookup)} demonstrations of skill {skill_name}.")
        return episode_lookup

import logging
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, Union, Tuple
from datasets.utils.load_utils import get_transforms

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Abstract datamodule base class.
    Args:

    """

    def __init__(
        self,
        data_dir: Path,
        skill: str,
        step_len: int,
        train: bool,
        transforms: Dict,
    ):
        self.data_dir = data_dir
        self.skill = skill
        self.step_len = step_len
        self.train = train
        self.transforms = transforms

        self.transform_robot_obs = None
        if "robot_obs" in self.transforms:
            self.transform_robot_obs = get_transforms(self.transforms.robot_obs)

    def __getitem__(self, idx: Union[int, Tuple[int, int]]) -> Dict:
        """
        Get sequence of datamodule.
        Args:
            idx: Index of the sequence.
        Returns:
            Loaded sequence.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns:
            Size of the datamodule.
        """
        # return len(self.episode_lookup)
        raise NotImplementedError

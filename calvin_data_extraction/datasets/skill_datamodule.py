import logging
from pathlib import Path
from typing import Dict

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class SkillDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        batch_size: int,
        num_workers: int,
        transforms: Dict,
        step_len: int,
        skill: str,
        **kwargs: Dict,
    ):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None

        self.dataset = dataset
        root_data_path = Path(self.dataset.data_dir).expanduser()
        # if not root_data_path.is_absolute():
        #     root_data_path = Path(__name__.__file__).parent / root_data_path

        self.train_dir = root_data_path / "training"
        self.val_dir = root_data_path / "validation"

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.step_len = step_len
        self.skill = skill

    def setup(self, stage=None):
        self.train_dataset = hydra.utils.instantiate(
            self.dataset,
            data_dir=self.train_dir,
            skill=self.skill,
            step_len=self.step_len,
            train=True,
            transforms=self.transforms,
        )
        self.val_dataset = hydra.utils.instantiate(
            self.dataset,
            data_dir=self.val_dir,
            skill=self.skill,
            step_len=self.step_len,
            train=False,
            transforms=self.transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

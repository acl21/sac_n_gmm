import logging
from pathlib import Path
import sys
from typing import List, Union
import os
import numpy as np

cwd_path = Path(__file__).absolute().parents[0]
parent_path = cwd_path.parents[0]
# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, parent_path.as_posix())

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
import pdb

logger = logging.getLogger(__name__)
os.chdir(cwd_path)


def save_demonstrations(datamodule, save_dir):
    mode = ["training", "validation"]
    p = Path(Path(save_dir).expanduser() / datamodule.skill)
    p.mkdir(parents=True, exist_ok=True)

    for m in mode:
        if "train" in m:
            data_loader = datamodule.train_dataloader()
        elif "val" in m:
            data_loader = datamodule.val_dataloader()

        split_iter = iter(data_loader)
        time_step = 0.005
        time = np.expand_dims(np.arange(64 / datamodule.step_len) * time_step, axis=1)
        demos = []
        for i in range(len(split_iter)):
            demo = next(split_iter)
            demo = np.concatenate(
                [np.repeat(time[np.newaxis, :, :], demo["robot_obs"].size(0), axis=0), demo["robot_obs"]], axis=2
            )
            demos += [demo]

        demos = np.concatenate(demos, axis=0)
        logger.info(f"Dimensions of {m} demonstrations (NxSxD): {demos.shape}.")
        save_dir = p / m
        np.save(save_dir, demos)


@hydra.main(config_path="./config", config_name="demos")
def extract_demos(cfg: DictConfig) -> None:
    """
    This is called to extract demonstrations for a specific skill.
    Args:
        cfg: hydra config
    """
    seed_everything(cfg.seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="fit")
    save_demonstrations(datamodule, cfg.demos_dir)


if __name__ == "__main__":
    extract_demos()
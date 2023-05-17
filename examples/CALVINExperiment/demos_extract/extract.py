import os
import sys
import hydra
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

cwd_path = Path(__file__).absolute().parents[0]
calvin_exp_path = cwd_path.parents[0]
root = calvin_exp_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, calvin_exp_path.as_posix())  # CALVINExperiment
sys.path.insert(0, root.as_posix())  # Root

logger = logging.getLogger(__name__)
os.chdir(cwd_path)


def save_demonstrations(datamodule, save_dir, skill):
    mode = ["training", "validation"]
    p = Path(Path(save_dir).expanduser() / skill)
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
                [
                    np.repeat(
                        time[np.newaxis, :, :], demo["robot_obs"].size(0), axis=0
                    ),
                    demo["robot_obs"],
                ],
                axis=2,
            )
            demos += [demo]

        demos = np.concatenate(demos, axis=0)
        logger.info(f"Dimensions of {m} demonstrations (NxSxD): {demos.shape}.")
        save_dir = p / m
        np.save(save_dir, demos)


@hydra.main(version_base="1.1", config_path="../../config", config_name="demos_extract")
def extract_demos(cfg: DictConfig) -> None:
    """
    This is called to extract demonstrations for a specific skill.
    Args:
        cfg: hydra config
    """
    seed_everything(cfg.seed, workers=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.demos_dir, exist_ok=True)
    f = open(cfg.skills_to_extract, "r")
    skill_set = f.read()
    skill_set = skill_set.split("\n")
    for skill in skill_set:
        cfg.skill = skill
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        datamodule.setup(stage="fit")
        save_demonstrations(datamodule, cfg.demos_dir, skill)


if __name__ == "__main__":
    extract_demos()

import logging
import hydra
import os
import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.utilities import rank_zero_only
from sac_n_gmm.utils.utils import print_system_env_info, setup_logger, setup_callbacks, get_last_checkpoint

cwd_path = Path(__file__).absolute().parents[0]  # scripts
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

sys.path.insert(0, sac_gmm_path.as_posix())
sys.path.insert(0, os.path.join(root, "calvin_env"))  # root/calvin_env
sys.path.insert(0, root.as_posix())

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


@hydra.main(version_base="1.1", config_path="../../config", config_name="sac_n_gmm_ft")
def eval(cfg: DictConfig) -> None:
    if cfg.exp_dir is None:
        cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    model_dir = Path(cfg.exp_dir) / "model_weights/"
    cfg.callbacks.checkpoint.dirpath = model_dir
    os.makedirs(model_dir, exist_ok=True)

    if cfg.seed is None:
        import numpy as np

        cfg.seed = np.random.randint(0, 10000)
        cfg.comment = f"{cfg.comment}.{cfg.seed}"
    seed_everything(cfg.seed, workers=True)

    log_rank_0(f"Training a SAC-GMM skill with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())
    chk = get_last_checkpoint(model_dir)

    agent = hydra.utils.instantiate(cfg.agent)
    algo = hydra.utils.instantiate(cfg.rl, agent=agent)

    algo.evaluate()


if __name__ == "__main__":
    eval()

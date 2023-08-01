"""Launch skill-based RL training."""

import hydra
from omegaconf import OmegaConf, DictConfig

from rolf.main import Run


class SkillRLRun(Run):
    def _set_run_name(self):
        """Sets run name."""
        cfg = self._cfg
        if "phase" in cfg.rolf:
            cfg.run_name = f"{cfg.env.id}.{cfg.rolf.name}.{cfg.rolf.phase}.{cfg.run_prefix}.{cfg.seed}"
        else:
            super()._set_run_name()

    def _get_trainer(self):
        if self._cfg.rolf.name in ["spirl_dreamer", "spirl_tdmpc", "skimo", "seqref"]:
            from skill_trainer import SkillTrainer

            return SkillTrainer(self._cfg)
        if self._cfg.rolf.name == "spirl":
            from spirl_trainer import SPiRLTrainer

            return SPiRLTrainer(self._cfg)
        return super()._get_trainer()


@hydra.main(config_path="config", config_name="seqref_calvin")
def main(cfg: DictConfig) -> None:
    # Debugging
    cfg.run_prefix = "debug"
    cfg.gpu = 0
    cfg.wandb = False
    cfg.rolf.phase = "rl"
    cfg.rolf.warm_up_step = 20
    # cfg.rolf.pretrain_ckpt_path = (
    #     "log/calvin.skimo.pretrain.test1.0/ckpt/ckpt_00000005000.pt"
    # )
    cfg.rolf.pretrain_ckpt_path = (
        "log/calvin.seqref.pretrain.debug.0/ckpt/ckpt_00000200000.pt"
    )

    # Make config writable
    OmegaConf.set_struct(cfg, False)

    # Change default config
    cfg.wandb_entity = "in-ac"
    cfg.wandb_project = "seqref"

    # Execute training code
    SkillRLRun(cfg).run()


if __name__ == "__main__":
    main()

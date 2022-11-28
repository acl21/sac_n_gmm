import os
import sys
from pathlib import Path

cwd_path = Path(__file__).absolute().parents[0]
parent_path = cwd_path.parents[0]

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, parent_path.as_posix())
sys.path.insert(0, cwd_path.parents[0].parents[0].as_posix()) # Root

import hydra
from omegaconf import DictConfig

import logging
import pdb

class SkillTrainer(object):
    """Python wrapper that allows you to train DS skills on a given dataset
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.state_type = self.cfg.state_type
        self.logger = logging.getLogger('SkillTrainer')
        self.ds_out_dir = None
        # Make skills directory if doesn't exist
        os.makedirs(self.cfg.skills_dir, exist_ok=True)

    def run(self):
        f = open(self.cfg.skills_list, "r")
        skill_set = f.read()
        skill_set = skill_set.split("\n")
        self.logger.info(f'Found {len(skill_set)} skills in the list')
        self.logger.info(f'Training DS with {self.state_type} as the input')
        for idx, skill in enumerate(skill_set):
            # Make output dir where trained models will be saved
            self.ds_out_dir = os.path.join(self.cfg.skills_dir, self.state_type, skill, self.cfg.ds_type)
            os.makedirs(self.ds_out_dir, exist_ok=True)
            # Load dataset
            self.cfg.dataset.skill = skill
            self.cfg.dataset.train = True
            train_dataset = hydra.utils.instantiate(self.cfg.dataset)
            self.cfg.dataset.train = False
            val_dataset = hydra.utils.instantiate(self.cfg.dataset)
            self.logger.info(f'Skill {idx}: {skill}, Train Data: {train_dataset.X.size()}, Val. Data: {val_dataset.X.size()}')
            self.cfg.dim = train_dataset.X.shape[-1]
            # Train DS
            if self.cfg.ds_type == 'clfds':
                clfds = hydra.utils.instantiate(self.cfg.dyn_sys)
                clfds.train_clf(train_dataset, val_dataset, lr=self.cfg.lr, max_epochs=self.cfg.max_epochs,\
                batch_size=self.cfg.batch_size, fname=os.path.join(self.ds_out_dir, 'clf'), wandb_flag=self.cfg.wandb)
                assert os.path.exists(os.path.join(self.ds_out_dir, 'clf')), f"CLF file not found at {os.path.join(self.ds_out_dir)}"
                clfds.load_clf_model(os.path.join(self.ds_out_dir, 'clf'))
                clfds.train_ds(train_dataset, val_dataset, lr=self.cfg.lr, max_epochs=self.cfg.max_epochs,\
                    batch_size=self.cfg.batch_size, fname=os.path.join(self.ds_out_dir, 'ds'), wandb_flag=self.cfg.wandb)
            else:
                gmm = hydra.utils.instantiate(self.cfg.dyn_sys)
                pdb.set_trace()
        self.logger.info(f'Training complete. Trained DS models are saved in the {os.path.join(self.ds_out_dir)} directory')

@hydra.main(config_path="./config", config_name="train_ds")
def main(cfg: DictConfig) -> None:
    eval = SkillTrainer(cfg)
    eval.run()

if __name__ == "__main__":
    main()
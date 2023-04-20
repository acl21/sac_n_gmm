import os
import sys
import csv
import hydra
import wandb
import torch
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from examples.CALVINExperiment.envs.skill_env import SkillSpecificEnv

cwd_path = Path(__file__).absolute().parents[0]
calvin_exp_path = cwd_path.parents[0]
root = calvin_exp_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, calvin_exp_path.as_posix()) # CALVINExperiment
sys.path.insert(0, os.path.join(calvin_exp_path, 'calvin_env')) # CALVINExperiment/calvin_env
sys.path.insert(0, root.as_posix()) # Root


class SkillEvaluator(object):
    """Python wrapper that allows you to evaluate learned DS skills 
    in the CALVIN environment.
    """

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        f = open(cfg.skills_list, "r")
        skill_set = f.read()
        self.skill_set = skill_set.split("\n")
        self.logger = logging.getLogger('SkillEvaluator')

    def evaluate(self, ds, dataset, max_steps, sampling_dt, render=False, record=False):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
        start_idx, end_idx = self.env.get_valid_columns()
        for idx, (xi, d_xi) in enumerate(dataloader):
            if (idx % 5 == 0) or (idx == len(dataset)):
                self.logger.info(f'Test Trajectory {idx+1}/{len(dataset)}')
            x0 = xi.squeeze()[0, :].numpy()
            goal = dataset.goal
            rollout_return = 0
            observation = self.env.reset()
            current_state = observation[start_idx:end_idx]
            if dataset.state_type == 'pos':
                temp = np.append(x0, np.append(dataset.fixed_ori, -1))
            else:
                temp = np.append(x0, -1)
            action = self.env.prepare_action(temp, type='abs')
            # self.logger.info(f'Adjusting EE position to match the initial pose from the dataset')
            count = 0
            error_margin = 0.01
            while np.linalg.norm(current_state - x0) >= error_margin:
                observation, reward, done, info = self.env.step(action)
                current_state = observation[start_idx:end_idx]
                count += 1
                if count >= 200:
                    # x0 = current_state
                    self.logger.info("CALVIN is struggling to place the EE at the right initial pose")
                    self.logger.info(x0, current_state, np.linalg.norm(current_state - x0))
                    break
            x = observation[start_idx:end_idx]
            # self.logger.info(f'Simulating with DS')
            if record:
                self.logger.info(f'Recording Robot Camera Obs')
                self.env.record_frame()
            for step in range(max_steps):
                if ds.name == 'clfds':
                    d_x = ds.reg_model.forward(torch.from_numpy(x-goal).float().unsqueeze(dim=0).unsqueeze(dim=0))
                    d_x = d_x.detach().cpu().numpy().squeeze()
                    delta_x = sampling_dt * d_x
                    new_x = x + delta_x
                else:
                    # goal = np.array([0.16425726, -0.23430869,  0.3718335])
                    # goal = np.array([0.17921756, -0.21936302,  0.38075492])
                    # First Goal-Centering and then Normalize (GCN Space)
                    # d_x = ds.predict_dx(dataset.normalize(x-goal))
                    d_x = ds.predict_dx(x-goal)
                    delta_x = sampling_dt * d_x
                    # Get next position in GCN space
                    # new_x = dataset.normalize(x-goal) + delta_x
                    new_x = x + delta_x
                    # Come back to original data space from GCN space
                    # new_x = dataset.undo_normalize(new_x) + goal
                    # pdb.set_trace()
                if dataset.state_type == 'pos':
                    temp = np.append(new_x, np.append(dataset.fixed_ori, -1))
                else:
                    temp = np.append(new_x, -1)
                action = self.env.prepare_action(temp, type='abs')
                observation, reward, done, info = self.env.step(action)
                x = observation[start_idx:end_idx]
                rollout_return += reward
                if record:
                    self.env.record_frame()
                if render:
                    self.env.render()
                if done:
                    break
            status = None
            if info["success"]:
                succesful_rollouts += 1
                status = 'Success'
            else:
                status = 'Fail'
            self.logger.info(f'{idx+1}: {status}!')
            if record:
                self.logger.info(f'Saving Robot Camera Obs')
                video_path = self.env.save_recorded_frames()
                self.env.reset_recorded_frames()
                status = None
                if self.cfg.wandb:
                    wandb.log({f"{self.env.skill_name} {status} {self.env.record_count}":wandb.Video(video_path, fps=30, format="gif")})
            rollout_returns.append(rollout_return)
            rollout_lengths.append(step)
        acc = succesful_rollouts / len(dataset.X)
        if self.cfg.wandb:
            wandb.config.update({'val dataset size': len(dataset.X)})
            wandb.log({'skill':self.env.skill_name, 'accuracy': acc*100, \
                       'average_return': np.mean(rollout_returns), \
                       'average_traj_len': np.mean(rollout_lengths)})
        return acc, np.mean(rollout_returns), np.mean(rollout_lengths)

    def run(self):
        skill_accs = {}
        for skill in self.skill_set:
            if self.cfg.wandb:
                config = {'state_type': self.cfg.state_type, \
                        'sampling_dt': self.cfg.sampling_dt, \
                        'max steps': self.cfg.max_steps}
                wandb.init(project="ds-evaluation", entity="in-ac", config=config, name=f'{skill}_{self.cfg.state_type}')
            self.env.set_skill(skill)

            # Get validation dataset
            self.cfg.dataset.skill = skill
            val_dataset = hydra.utils.instantiate(self.cfg.dataset)

            # Create and load models to evaluate
            self.cfg.dim = val_dataset.X.shape[-1]
            ds = hydra.utils.instantiate(self.cfg.dyn_sys)
            ds_model_dir = os.path.join(self.cfg.skills_dir, self.cfg.state_type, skill, ds.name)

            if ds.name == 'clfds':
                clf_file = os.path.join(ds_model_dir, 'clf')
                reg_file = os.path.join(ds_model_dir, 'ds')
                ds.load_models(clf_file=clf_file, reg_file=reg_file)
            else:
                # Obtain X_mins and X_maxs from training data to normalize in real-time
                self.cfg.dataset.train = True
                # self.cfg.dataset.normalized = True
                # self.cfg.dataset.goal_centered = True
                train_dataset = hydra.utils.instantiate(self.cfg.dataset)
                val_dataset.goal = train_dataset.goal
                # val_dataset.X_mins = train_dataset.X_mins
                # val_dataset.X_maxs = train_dataset.X_maxs

                ds.skills_dir = ds_model_dir
                ds.load_params()
                ds.state_type = self.cfg.state_type
                ds.manifold = ds.make_manifold(self.cfg.dim)
            self.logger.info(f'Evaluating {skill} skill with {self.cfg.state_type} input on CALVIN environment')
            self.logger.info(f'Test/Val Data: {val_dataset.X.size()}')
            # Evaluate by simulating in the CALVIN environment
            acc, avg_return, avg_len = self.evaluate(ds, val_dataset, max_steps=self.cfg.max_steps, \
                                                    render=self.cfg.render, record=self.cfg.record, \
                                                    sampling_dt=self.cfg.sampling_dt)
            skill_accs[skill] = [str(acc), str(avg_return), str(avg_len)]
            self.env.count = 0
            if self.cfg.wandb:
                wandb.finish()

            # Log evaluation output
            self.logger.info(f'{skill} Skill Accuracy: {round(acc, 2)}')

        # Write accuracies to a file
        with open(os.path.join(self.env.outdir, f'skill_ds_acc_{self.cfg.state_type}.txt'), 'w') as f:
            writer = csv.writer(f)
            for row in skill_accs.items():
                writer.writerow(row)


@hydra.main(version_base='1.1', config_path="../config", config_name="eval_ds")
def main(cfg: DictConfig) -> None:
    new_env_cfg = {**cfg.calvin_env.env}
    new_env_cfg["use_egl"] = False
    new_env_cfg["show_gui"] = False
    new_env_cfg["use_vr"] = False
    new_env_cfg["use_scene_info"] = True
    new_env_cfg["tasks"] = cfg.calvin_env.tasks
    new_env_cfg.pop("_target_", None)
    new_env_cfg.pop("_recursive_", None)

    env = SkillSpecificEnv(**new_env_cfg)
    env.set_state_type(cfg.state_type)
    env.set_outdir(hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir'])

    eval = SkillEvaluator(cfg, env)
    eval.run()

if __name__ == "__main__":
    main()
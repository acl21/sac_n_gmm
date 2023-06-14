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
sys.path.insert(0, calvin_exp_path.as_posix())  # CALVINExperiment
sys.path.insert(
    0, os.path.join(calvin_exp_path, "calvin_env")
)  # CALVINExperiment/calvin_env
sys.path.insert(0, root.as_posix())  # Root


def sample_start(start, size=1, sigma=0.15):
    sampled = sample_gaussian_norm_ball(start, sigma, size)
    if size == 1:
        return sampled[0]
    else:
        return sampled


def sample_gaussian_norm_ball(reference_point, sigma, num_samples):
    samples = []
    for _ in range(num_samples):
        # Step 1: Sample from standard Gaussian distribution
        offset = np.random.randn(3)

        # Step 2: Normalize the offset
        normalized_offset = offset / np.linalg.norm(offset)

        # Step 3: Scale the normalized offset
        scaled_offset = normalized_offset * np.random.normal(0, sigma)

        # Step 4: Translate the offset
        sampled_point = reference_point + scaled_offset

        samples.append(sampled_point)

    return samples


def calibrate_EE_start_state(obs, env, dataset, error_margin=0.01, max_checks=15):
    """Samples a random starting point and moves the end effector to that point"""
    desired_start = sample_start(dataset.start, size=1, sigma=0.05)
    count = 0
    state = np.append(desired_start, np.append(dataset.fixed_ori, -1))
    action = env.prepare_action(state, type="abs")
    while np.linalg.norm(obs[0:3] - desired_start) > error_margin:
        obs, _, _, _ = env.step(action)
        count += 1
        if count >= max_checks:
            # self.cons_logger.info(
            #     f"CALVIN is struggling to place the EE at the right initial pose. \
            #         Difference: {np.linalg.norm(obs - desired_start)}"
            # )
            break
    return obs


class SkillEvaluator(object):
    """Python wrapper that allows you to evaluate learned DS skills
    in the CALVIN environment.
    """

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.skill = self.cfg.skill
        self.logger = logging.getLogger("SkillEvaluator")

    def evaluate(self, ds, dataset, max_steps, sampling_dt, render=False, record=False):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
        start_idx, end_idx = self.env.get_valid_columns()
        for idx, (xi, d_xi) in enumerate(dataloader):
            if idx in [0, 15, 18, 24, 26, 29]:
                continue
            if (idx % 5 == 0) or (idx == len(dataset)):
                self.logger.info(f"Test Trajectory {idx+1}/{len(dataset)}")
            goal = dataset.goal
            rollout_return = 0
            observation = self.env.reset()
            # observation = calibrate_EE_start_state(
            #     observation[start_idx:end_idx], self.env, dataset
            # )
            desired_start = xi[0, 0, :3].numpy()
            max_checks = 20
            error_margin = 0.01
            count = 0
            state = np.append(desired_start, np.append(dataset.fixed_ori, -1))
            action = self.env.prepare_action(state, type="abs")
            while np.linalg.norm(observation[0:3] - desired_start) > error_margin:
                observation, _, _, _ = self.env.step(action)
                count += 1
                if count >= max_checks:
                    self.logger.info(
                        f"CALVIN is struggling to place the EE at the right initial pose. \
                            Difference: {np.linalg.norm(observation[0:3] - desired_start)}"
                    )
                    break

            x = observation[start_idx:end_idx]
            # self.logger.info(f'Simulating with DS')
            if record:
                self.logger.info("Recording Robot Camera Obs")
                self.env.record_frame()
            for step in range(max_steps):
                if ds.name == "clfds":
                    d_x = ds.reg_model.forward(
                        torch.from_numpy(x - goal)
                        .float()
                        .unsqueeze(dim=0)
                        .unsqueeze(dim=0)
                    )
                    d_x = d_x.detach().cpu().numpy().squeeze()
                    delta_x = sampling_dt * d_x
                    new_x = x + delta_x
                else:
                    # goal = np.array([0.16425726, -0.23430869,  0.3718335])
                    # goal = np.array([0.17921756, -0.21936302,  0.38075492])
                    # First Goal-Centering and then Normalize (GCN Space)
                    # d_x = ds.predict_dx(dataset.normalize(x-goal))
                    d_x = ds.predict_dx(x - goal)
                    # print(d_x)
                    delta_x = sampling_dt * d_x
                    # Get next position in GCN space
                    # new_x = dataset.normalize(x-goal) + delta_x
                    new_x = x + delta_x
                    # Come back to original data space from GCN space
                    # new_x = dataset.undo_normalize(new_x) + goal
                    # pdb.set_trace()
                if dataset.state_type == "pos":
                    temp = np.append(new_x, np.append(dataset.fixed_ori, -1))
                else:
                    temp = np.append(new_x, -1)
                action = self.env.prepare_action(temp, type="abs")
                observation, reward, done, info = self.env.step(action)
                x = observation[start_idx:end_idx]
                rollout_return += reward
                # Premature exit when close to skill's goal
                dist_to_goal = np.linalg.norm(
                    observation[start_idx:end_idx] - dataset.goal
                )
                # print(step, np.round(dist_to_goal, 2), info["success"])
                if record:
                    self.env.record_frame()
                if render:
                    self.env.render()
                # if done:
                #     break
                if np.round(dist_to_goal, 2) <= 0.01:
                    break
            status = None
            if info["success"]:
                # current_info = self.env.get_info()
                # print(
                #     current_info["scene_info"]["buttons"]["base__button"]["joint_state"]
                # )
                succesful_rollouts += 1
                status = "Success"
            else:
                status = "Fail"
            self.logger.info(f"{idx+1}: {status}!")
            if record:
                self.logger.info("Saving Robot Camera Obs")
                video_path = self.env.save_recorded_frames()
                self.env.reset_recorded_frames()
                status = None
                if self.cfg.wandb:
                    wandb.log(
                        {
                            f"{self.env.skill_name} {status} {self.env.record_count}": wandb.Video(
                                video_path, fps=30, format="gif"
                            )
                        }
                    )
            rollout_returns.append(rollout_return)
            rollout_lengths.append(step)
        acc = succesful_rollouts / len(dataset.X)
        if self.cfg.wandb:
            wandb.config.update({"val dataset size": len(dataset.X)})
            wandb.log(
                {
                    "skill": self.env.skill_name,
                    "accuracy": acc * 100,
                    "average_return": np.mean(rollout_returns),
                    "average_traj_len": np.mean(rollout_lengths),
                }
            )
        return acc, np.mean(rollout_returns), np.mean(rollout_lengths)

    def run(self):
        if self.cfg.wandb:
            config = {
                "state_type": self.cfg.state_type,
                "sampling_dt": self.cfg.sampling_dt,
                "max steps": self.cfg.max_steps,
            }
            wandb.init(
                project="ds-evaluation",
                entity="in-ac",
                config=config,
                name=f"{self.skill}_{self.cfg.state_type}",
            )
        self.env.set_skill(self.skill)

        # Get validation dataset
        self.cfg.dataset.skill = self.skill
        self.cfg.dataset.train = False
        val_dataset = hydra.utils.instantiate(self.cfg.dataset)

        # Create and load models to evaluate
        self.cfg.dim = val_dataset.X.shape[-1]
        ds = hydra.utils.instantiate(self.cfg.dyn_sys)
        ds_model_dir = os.path.join(
            self.cfg.skills_dir, self.cfg.state_type, self.skill, ds.name
        )

        if ds.name == "clfds":
            clf_file = os.path.join(ds_model_dir, "clf")
            reg_file = os.path.join(ds_model_dir, "ds")
            ds.load_models(clf_file=clf_file, reg_file=reg_file)
        else:
            # Obtain X_mins and X_maxs from training data to normalize in real-time
            self.cfg.dataset.train = True
            # self.cfg.dataset.normalized = True
            # self.cfg.dataset.goal_centered = True
            train_dataset = hydra.utils.instantiate(self.cfg.dataset)
            val_dataset.goal = train_dataset.goal
            val_dataset.fixed_ori = train_dataset.fixed_ori
            # val_dataset.X_mins = train_dataset.X_mins
            # val_dataset.X_maxs = train_dataset.X_maxs

            ds.skills_dir = ds_model_dir
            ds.load_params()
            ds.state_type = self.cfg.state_type
            ds.manifold = ds.make_manifold(self.cfg.dim)
        self.logger.info(
            f"Evaluating {self.skill} skill with {self.cfg.state_type} input on CALVIN environment"
        )
        self.logger.info(f"Test/Val Data: {val_dataset.X.size()}")
        # Evaluate by simulating in the CALVIN environment
        acc, avg_return, avg_len = self.evaluate(
            ds,
            val_dataset,
            max_steps=self.cfg.max_steps,
            render=self.cfg.render,
            record=self.cfg.record,
            sampling_dt=self.cfg.dt,
        )

        self.env.count = 0
        if self.cfg.wandb:
            wandb.finish()

        # Log evaluation output
        self.logger.info(f"{self.skill} Skill Accuracy: {round(acc, 2)}")


@hydra.main(version_base="1.1", config_path="../config", config_name="eval_ds")
def main(cfg: DictConfig) -> None:
    env = SkillSpecificEnv(**cfg.calvin_env)
    env.set_state_type(cfg.state_type)
    env.set_outdir(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])

    eval = SkillEvaluator(cfg, env)
    eval.run()


if __name__ == "__main__":
    main()

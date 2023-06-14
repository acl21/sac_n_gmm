import os
import sys
import hydra
import wandb
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from examples.CALVINExperiment.envs.task_env import TaskSpecificEnv

cwd_path = Path(__file__).absolute().parents[0]
calvin_exp_path = cwd_path.parents[0]
root = calvin_exp_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, calvin_exp_path.as_posix())  # CALVINExperiment
sys.path.insert(
    0, os.path.join(calvin_exp_path, "calvin_env")
)  # CALVINExperiment/calvin_env
sys.path.insert(0, root.as_posix())  # Root


class TaskDemoEvaluator(object):
    """Python wrapper that allows you to evaluate learned DS skills
    in the CALVIN environment.
    """

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.task = cfg.target_tasks
        self.task_name = "_".join(self.task)
        self.logger = logging.getLogger("TaskDemoEvaluator")

    def evaluate(self, dataset, max_steps, sampling_dt, render=False, record=False):
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        succesful_rollouts, rollout_returns, rollout_lengths = 0, [], []
        start_idx, end_idx = 0, 3
        for idx, (xi, d_xi) in enumerate(dataloader):
            if (idx % 5 == 0) or (idx == len(dataset)):
                self.logger.info(f"Test Trajectory {idx+1}/{len(dataset)}")
            x0 = xi.squeeze()[0, :].numpy()
            rollout_return = 0
            observation = self.env.reset()
            current_state = observation
            if dataset.state_type == "pos":
                x = np.append(x0, np.append(dataset.fixed_ori, -1))
            else:
                x = np.append(x0, -1)
            action = self.env.prepare_action(x, type="abs")
            # self.logger.info(f'Adjusting EE position to match the initial pose from the dataset')
            count = 0
            error_margin = 0.01
            while np.linalg.norm(current_state[:3] - x0) >= error_margin:
                observation, reward, done, info = self.env.step(action)
                current_state = observation
                count += 1
                if count >= 200:
                    self.logger.info(
                        "CALVIN is struggling to place the EE at the right initial pose"
                    )
                    self.logger.info(
                        x0, current_state[:3], np.linalg.norm(current_state[:3] - x0)
                    )
                    break
            # pdb.set_trace()
            # self.logger.info(f'Simulating with DS')
            if record:
                self.logger.info("Recording Robot Camera Obs")
                self.env.record_frame()
            for step in range(len(xi.squeeze())):
                new_x = xi.squeeze()[step, :]
                if dataset.state_type == "pos":
                    new_x = np.append(new_x, np.append(dataset.fixed_ori, -1))
                else:
                    new_x = np.append(new_x, -1)
                action = self.env.prepare_action(new_x, type="abs")
                observation, reward, done, info = self.env.step(action)
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
                status = "Success"
            else:
                status = "Fail"
            self.logger.info(f"{idx+1}: {status}!")
            if record:
                self.logger.info("Saving Robot Camera Obs")
                video_path = self.env.save_recorded_frames(
                    self.cfg.exp_dir, f"{status}"
                )
                self.env.reset_recorded_frames()
                status = None
                if self.cfg.wandb:
                    wandb.log(
                        {
                            f"{self.task_name} {status} {self.env.record_count}": wandb.Video(
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
                    "task": self.task_name,
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
                project="ds-evaluation-task-demos",
                entity="in-ac",
                config=config,
                name=f"{self.task_name}_{self.cfg.state_type}",
            )

        # Get validation dataset
        val_dataset = hydra.utils.instantiate(self.cfg.dataset)

        self.logger.info(
            f"Evaluating {self.task_name} task with {self.cfg.state_type} input on CALVIN environment"
        )
        self.logger.info(f"Test/Val Data: {val_dataset.X.size()}")
        # Create and load models to evaluate
        # Evaluate by simulating in the CALVIN environment
        acc, avg_return, avg_len = self.evaluate(
            val_dataset,
            max_steps=self.cfg.max_steps,
            render=self.cfg.render,
            record=self.cfg.record,
            sampling_dt=self.cfg.sampling_dt,
        )
        self.env.count = 0
        if self.cfg.wandb:
            wandb.finish()

        # Log evaluation output
        self.logger.info(f"{self.task_name} Task Accuracy: {round(acc, 2)}")


@hydra.main(version_base="1.1", config_path="../config", config_name="eval_task")
def main(cfg: DictConfig) -> None:
    new_env_cfg = {**cfg.calvin_env}
    new_env_cfg["target_tasks"] = cfg.target_tasks
    new_env_cfg["sequential"] = cfg.task_sequential

    env = TaskSpecificEnv(**new_env_cfg)
    env.state_type = cfg.state_type
    cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]

    eval = TaskDemoEvaluator(cfg, env)
    eval.run()


if __name__ == "__main__":
    main()

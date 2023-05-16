import os
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
from SkillsRefining.utils.utils import prepare_torch
from examples.CALVINExperiment.seqblend.CALVINSkill import CALVINSkill

device = prepare_torch()


class CALVINExp:
    """
    This class defines an experiment for a robot arm with a gripper in CALVIN environment.
    """
    def __init__(self, skill_names, demos_dir, skills_dir):
        """
        Initialization of the experiment class.

        Parameters
        ----------
        :param skill_list: list of skills
        :param skill_fcns: skills functions returning the desired control values for the list of skills

        """
        self.skill_names = skill_names
        self.skill_list = [CALVINSkill(skill, i, demos_dir, skills_dir) for i, skill in enumerate(self.skill_names)]
        self.skill_fncs = [self.get_ds_skill(skill) for skill in self.skill_list]
        self.demos_dir = demos_dir
        self.logger = logging.getLogger('CALVINExp')

    def get_ds_skill(self, skill):
        sampling_dt = 2/30

        def ds_skill(x):
            x = x[:3]
            d_x = skill.predict_dx(x)
            delta_x = sampling_dt * d_x
            x = x + delta_x
            return x

        return ds_skill

    def get_taskspace_training_data(self, fname='training.npy'):
        """
        This function returns the training data for a demonstration trajectory. The training data consists of the list
        of skills, the position and velocity along the trajectory, the desired skill values along the trajectory.

        Parameters
        ----------
        :param fname: file containing demonstration trajectory

        Return
        ------
        :return skill_list: list of skills
        :return xt_d: positions of the different skills along the demonstration
        :return dxt_d: velocities of the different skills along the demonstration
        :return desired_: desired control values given by the skills along the trajectory
        """
        task_dir = '_'.join(self.skill_names)
        self.demo_log = np.load(os.path.join(self.demos_dir, task_dir, fname))

        if self.demo_log is None:
            raise ValueError('Please generate demo first')
        else:
            self.logger.info(f'Loaded task demo data from {os.path.join(self.demos_dir, task_dir, fname)}')

        dt = 2/30
        X = self.demo_log[0]
        dX = (X[2:, :] - X[:-2, :]) / dt
        X = X[1:-1, :]
        for i in range(len(self.skill_list)):
            desired_value = np.stack([self.skill_fncs[i](X[t, :]) for t in np.arange(0, X.shape[0])])
            self.skill_list[i].desired_value = desired_value

        Xt = []
        dXt = []
        desired_ = []
        for i in range(len(self.skill_list)):
            Xt.append(X)
            dXt.append(dX)
            desired_.append(self.skill_list[i].desired_value)

        desired_ = np.hstack(desired_)
        Xt_d = np.hstack(Xt)
        dXt_d = np.hstack(dXt)

        self.logger.info(f'Returning task space training data!')
        return Xt_d, dXt_d, desired_

    def test_policy(self, policy, cfg, env):
        """
        Test a given policy. The robot is controlled in cartesian space.

        Parameters
        ---------
        :param policy: trained policy
        :param cfg: Hydra config
        :param env: CALVIN environment

        Return
        ------
        :return: resulting trajectory
        """
        if self.demo_log is None:
            task_dir = '_'.join(self.skill_names)
            self.demo_log = np.load(os.path.join(self.demos_dir, task_dir, fname='training.npy'))

        dt = 2/30
        X = self.demo_log[0]
        dX = (X[2:, :] - X[:-2, :]) / dt
        x0 = X[0, :]
        X = X[1:-1, :]

        timesteps = X.shape[0]
        timestamps = np.linspace(0, 1, timesteps)
        Xt_track = np.zeros((timesteps, 6))
        dXt_track = np.zeros((timesteps, 6))
        wmat_track = np.zeros((timesteps, sum([skill.dim() for skill in self.skill_list])))
        wmat_full_track = np.zeros((timesteps,
                                    sum([skill.dim() for skill in self.skill_list]),
                                    sum([skill.dim() for skill in self.skill_list])))
        
        self.logger.info('Testing Phase: Start Generating Trajectory ...')
        Xt = np.hstack((X, dX))

        start_idx, end_idx = 0, 3
        x0 = np.hstack((x0, x0))
        temp = np.append(x0[:3], np.append(self.skill_list[0].dataset.fixed_ori, -1))
        action = env.prepare_action(temp, type='abs')
        count = 0
        error_margin = 0.01
        observation = env.reset()
        current_state = observation[start_idx:end_idx]
        while np.linalg.norm(current_state - x0[:3]) >= error_margin:
            observation, reward, done, info = env.step(action)
            current_state = observation[start_idx:end_idx]
            count += 1
            if count >= 200:
                self.logger.info("CALVIN is struggling to place the EE at the right initial pose")
                self.logger.info(x0[:3], current_state, np.linalg.norm(current_state - x0[:3]))
                break
        xt = np.hstack((current_state[start_idx:end_idx], x0[3:]))

        if cfg.record:
            self.logger.info(f'Recording Robot Camera Obs')
            env.reset_recorded_frames()
            env.record_frame()

        for i in range(timesteps):
            self.logger.info('Timestamp %1d / %1d' % (i, timesteps))
            [self.skill_list[s].update_desired_value(self.skill_fncs[s](xt[:3])) for s in range(len(self.skill_list))]

            desired_ = []
            for skill in self.skill_list:
                desired_value = torch.from_numpy(skill.desired_value).double().to(device)
                desired_.append(desired_value)

            desired_ = torch.cat(desired_, dim=-1)
            feat = torch.from_numpy(np.array([timestamps[i]])).double().to(device)
            Xt_input = torch.from_numpy(xt).double().to(device)
            Xt_input = torch.unsqueeze(Xt_input, 0)
            dxt, wmat, ddata = policy.forward(feat, Xt_input, desired_)
            
            # Simple check to switch between the two skills
            # if ddata.detach().cpu().numpy()[0, 0] > 0.5:
            #     xt_weighted = xt[:3]
            # else:
            #     xt_weighted = xt[3:]

            # Weighted sum of velocities
            dxt = np.hstack((self.skill_list[0].predict_dx(xt[:3]), self.skill_list[1].predict_dx(xt[:3])))
            dxt_weighted = ddata.detach().cpu().numpy() * dxt
            dxt_weighted = dxt_weighted[:, :3] + dxt_weighted[:, 3:]
            xt_weighted = xt[:3] + dxt_weighted[0] * dt

            # Weighted sum of the two next positions (doesn't work well)
            # dxt = np.hstack((self.skill_list[0].predict_dx(xt[:3]), self.skill_list[1].predict_dx(xt[:3])))
            # dxt = dxt.detach().cpu().numpy()
            # xt1 = dxt[:, :3] * dt  
            # xt2 = dxt[:, 3:] * dt
            # xt_weighted = ddata.detach().cpu().numpy()[0, 0] * xt1 + (1 - ddata.detach().cpu().numpy()[0, 0]) * xt2

            # Act in the environment
            temp = np.append(xt_weighted, np.append(self.skill_list[0].dataset.fixed_ori, -1))
            action = env.prepare_action(temp, type='abs')
            observation, reward, done, info = env.step(action)
            xt = np.hstack((observation[start_idx:end_idx], observation[start_idx:end_idx]))

            # Track some values of interest
            Xt_track[i, :] = xt
            dXt_track[i, :] = dxt
            wmat_track[i, :] = ddata.detach().cpu().numpy()
            wmat_full_track[i, :, :] = wmat.detach().cpu().numpy()

            if cfg.record:
                env.record_frame()
            if cfg.render:
                env.render()
            if info["success"]:
                self.logger.info('Success')
            if done:
                break

        if cfg.record:
            self.logger.info(f'Saving Robot Camera Obs')
            video_path = env.save_recorded_frames()
            env.reset_recorded_frames()
            status = None

        res = {'Xt_track': Xt_track,
               'dXt_track': dXt_track,
               'wmat_track': wmat_track}
        
        if cfg.plot:
            # Line plot of all the weights
            plt.figure()
            plt.plot(wmat_full_track[:, 0, 0], label='w_open')
            plt.legend()
            plt.xlabel('Timesteps')
            plt.ylabel('Weights')
            plt.title('Weighting of the two skills')
            plt.savefig(os.path.join(env.outdir, 'weights.png'))
            plt.show()

            # 3D plot of the trajectory
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(Xt_track[:, 0], Xt_track[:, 1], Xt_track[:, 2], label='Policy')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], label='GT')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.view_init(10, -130)
            plt.title('Trajectory in 3D')
            plt.legend()
            plt.savefig(os.path.join(env.outdir, 'traj.png'))
            plt.show()

if __name__ == "__main__":
    exp = CALVINExp(skill_names=['open_drawer', 'close_drawer'])
    Xt_d, dXt_d, desired_ = exp.get_taskspace_training_data()
    print(Xt_d.shape, dXt_d.shape, desired_.shape)
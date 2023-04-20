import os
import torch
import logging
import numpy as np
from SkillsSequencing.utils.utils import prepare_torch
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

    def test_policy(self, policy, fname=None, is_plot=False):
        """
        Test a given policy. The robot is controlled in cartesian space.

        Parameters
        ---------
        :param policy: trained policy

        Optional parameters
        -------------------
        :param is_plot: if True plot the resulting trajectory
        :param fname: demonstrations file
        :param is_plot_env: if True, display the resulting robot motion in CALVIN environment

        Return
        ------
        :return: resulting trajectory
        """
        if self.demo_log is None and fname is None:
            raise ValueError('Please generate demo first or give the demo log filename (.npz)')

        if fname is not None:
            demo_log = np.load(fname)
            self.demo_log = dict(demo_log)
        
        dt = 2/30
        X = self.demo_log[0]
        dX = (X[2:, :] - X[:-2, :]) / dt
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
        Xt = X[0, :]
        for i in range(timesteps):
            self.logger.info('Timestamp %1d / %1d' % (i, timesteps), end='\r')
            [self.skill_list[si].update_desired_value(self.skill_fncs[si](Xt)) for si in range(len(self.skill_list))]

            desired_ = []
            for skill in self.skill_list:
                desired_value = torch.from_numpy(skill.desired_value).double().to(device)
                desired_.append(desired_value)

            desired_ = torch.cat(desired_, dim=-1)
            feat = torch.from_numpy(np.array([timestamps[i]])).double().to(device)
            Xt_input = torch.from_numpy(Xt).double().to(device)
            Xt_input = torch.unsqueeze(Xt_input, 0)

            dXt, wmat, ddata = policy.forward(feat, Xt_input, desired_)
            dXt = dXt.detach().cpu().numpy()[0, :] * dt
            dXt_track[i, :] = dXt
            Xt = Xt + dXt
            Xt_track[i, :] = Xt
            wmat_track[i, :] = ddata.detach().cpu().numpy()
            wmat_full_track[i, :, :] = wmat.detach().cpu().numpy()

        res = {'Xt_track': Xt_track,
               'dXt_track': dXt_track,
               'wmat_track': wmat_track}

if __name__ == "__main__":
    exp = CALVINExp(skill_names=['open_drawer', 'close_drawer'])
    Xt_d, dXt_d, desired_ = exp.get_taskspace_training_data()
    print(Xt_d.shape, dXt_d.shape, desired_.shape)
import os
import numpy as np
from pymanopt.manifolds import Euclidean, Product
from SkillsSequencing.skills.mps.gmr.manifold_gmr import manifold_gmr
from SkillsSequencing.skills.mps.dynsys.CALVIN_DS import CALVINDynSysDataset

class CALVINSkill:
    """
    This class handles all functions related to CALVIN skills
    """
    def __init__(self, skill_name, skill_id, skills_dir=None):
        skills_dir = 'E:/Uni-Freiburg/Research/refining-skill-sequences/examples/CALVINExperiment/skills_ds/pos'
        demos_dir = 'E:/Uni-Freiburg/Research/refining-skill-sequences/examples/CALVINExperiment/data/'
        self.skill_name = skill_name
        self.skills_dir = skills_dir
        self.dim_ = 3
        self.state_idx = [i for i in range(skill_id*self.dim_, (skill_id+1)*self.dim_)]
        self.orignal_state_idx = self.state_idx

        self.means = None
        self.covariances = None
        self.priors = None
        self.load_skill_ds_params()

        self.manifold = Product([Euclidean(self.dim_), Euclidean(self.dim_)])
        self.dataset = CALVINDynSysDataset(skill=skill_name, train=True, state_type='pos',
                                           goal_centered=False, normalized=False,
                                           demos_dir=demos_dir)
        self.goal = self.dataset.goal
        self.desired_value = None


    def load_skill_ds_params(self):
        skill_file = f'{self.skills_dir}/{self.skill_name}/gmm/gmm_params.npz'

        if not os.path.exists(skill_file):
            raise FileNotFoundError(f'Skill GMM Params not found at {skill_file}')
        else: 
            print(f'Loading GMM params from {skill_file}')
        gmm = np.load(skill_file)
        gmm.allow_pickle = True
        self.means = np.array(gmm['gmm_means'])
        self.covariances = np.array(gmm['gmm_covariances'])
        self.priors = np.array(gmm['gmm_priors'])


    def predict_dx(self, x):
        dx, _, __ = manifold_gmr((x-self.goal).reshape(1, -1),
                                 self.manifold, self.means, 
                                 self.covariances, self.priors)

        return dx[0]

    def update_desired_value(self, desired_value, use_state_idx=False):
        if use_state_idx and self.state_idx is not None:
            desired_value = desired_value[:, self.state_idx]

        self.desired_value = desired_value

    def error(self, x, use_state_idx=False):
        if self.state_idx is not None and use_state_idx:
                x = x[:, self.state_idx]
        else:
            x = x[:, self.orignal_state_idx]

        return self.desired_value[:, :self.dim_] - x[:, :self.dim_]

    def dim(self):
        return self.dim_


class CALVINSkillComplex:
    """
    This class encodes a set of skills and allows the computation of the error including all skills in
    combined matrices.
    """
    def __init__(self, state_dim, config_dim, skills, skill_cluster_idx=None):
        """
        Initialization of the class
        :param state_dim: sum of dimensions of all skills
        :param config_dim: sum of dimensions of control variable (task-space dimension)
        :param skills: list of skills
        """
        self.skill_cluster_idx = skill_cluster_idx
        self.state_dim = state_dim
        self.config_dim = config_dim
        self.skills = skills
        self.n_skills = len(skills)
        self.skills_dim = []
        total_dim = 0
        for i in range(len(skills)):
            self.skills_dim.append(skills[i].dim())
            total_dim += skills[i].dim()

        self.total_dim = total_dim

    def update_desired_value(self, desired_value, use_state_idx=True):
        [skill.update_desired_value(desired_value, use_state_idx) for skill in self.skills]

    def error(self, state, use_state_idx=True):
        err = [skill.error(state, use_state_idx) for skill in self.skills]
        err = np.concatenate(err, axis=-1)
        return err
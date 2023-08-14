import numpy as np
import gym


def get_ref_param_space(cfg):
    """Returns GMM refining parameters range as a gym.spaces.Dict
    This helps in configuring high level agent's action space

    Returns:
        param_space : gym.spaces.Dict
            Range of GMM parameters parameters
    """
    # TODO: make low and high config variables
    dim = 3
    empty_priors = np.empty(shape=(cfg.pretrain.gmm_components,))
    empty_means = np.empty(shape=(cfg.pretrain.gmm_components, 2, dim))
    param_space = {}
    param_space["priors"] = gym.spaces.Box(
        low=-0.1, high=0.1, shape=(empty_priors.size,)
    )
    param_space["means"] = gym.spaces.Box(
        low=-0.01, high=0.01, shape=(empty_means.size,)
    )

    dim = empty_means.shape[1] // 2
    num_gaussians = empty_means.shape[0]
    cov_change_size = int(
        num_gaussians * dim * (dim + 1) / 2 + dim * dim * num_gaussians
    )
    param_space["covariances"] = gym.spaces.Box(
        low=-1e-6, high=1e-6, shape=(cov_change_size,)
    )
    return gym.spaces.Dict(param_space)


def get_meta_ac_space(cfg):
    """Returns high level agent's action space as gym.spaces.Dict
    Note that the high level agent must predict:
    - One hot vector of skill_dim size for sequencing
    - GMM change to refine gmms
    """
    # action_space = {"seq": gym.spaces.Discrete(cfg.skill_dim)}
    action_space = None
    refine_scale_max = 0.05  # Max refine change will only move the means by max 5 cm
    if cfg.refine.do:
        ref_param_space = get_ref_param_space(cfg)
        means_high = refine_scale_max * np.ones(ref_param_space["means"].shape[0])
        if cfg.refine.mean_shift:
            ref_action_high = means_high
        else:
            priors_high = refine_scale_max * np.ones(ref_param_space["priors"].shape[0])
            ref_action_high = np.concatenate((priors_high, means_high), axis=-1)
            if cfg.refine.adapt_cov:
                cov_high = np.ones(ref_param_space["covariances"].shape[0])
                ref_action_high = np.concatenate((ref_action_high, cov_high), axis=-1)

        ref_action_low = -ref_action_high
        action_space = gym.spaces.Box(ref_action_low, ref_action_high)
    return action_space


def get_refine_dict(cfg, gmm_change):
    """Takes refine vector and returns a dictionary for easy gmm update"""
    ref_param_space = get_ref_param_space(cfg)
    size_means = ref_param_space["means"].shape[0]
    dim = 3
    empty_means = np.empty(shape=(cfg.pretrain.gmm_components, 2, dim))
    if cfg.refine.mean_shift:
        # TODO: check low and high here
        means = np.hstack(
            [gmm_change.reshape((size_means, 1)) * ref_param_space["means"].high]
            # * empty_means.shape[1] TODO: Check this, removing this gives the right shape
        )

        refine_dict = {"means": means}
    else:
        size_priors = ref_param_space["priors"].shape[0]

        priors = gmm_change[:size_priors] * ref_param_space["priors"].high
        means = (
            gmm_change[size_priors : size_priors + size_means]
            * ref_param_space["means"].high
        )

        refine_dict = {"means": means, "priors": priors}
        if cfg.refine.adapt_cov:
            refine_dict["covariances"] = (
                gmm_change[size_priors + size_means :]
                * ref_param_space["covariances"].high
            )
    return refine_dict

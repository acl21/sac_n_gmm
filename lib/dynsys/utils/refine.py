import numpy as np
import gym


def get_ref_param_space(cfg):
    """Returns GMM refining parameters range as a gym.spaces.Dict
    This helps in configuring high level agent's action space

    Returns:
        param_space : gym.spaces.Dict
            Range of GMM parameters parameters
    """
    priors_change_range = 0.0
    mu_change_range = 0.05

    priors_size = cfg.pretrain.gmm_components
    means_size = cfg.pretrain.gmm_components * 2 * 3
    # TODO: make low and high config variables
    param_space = {}
    # param_space["priors"] = gym.spaces.Box(
    #     low=-priors_change_range,
    #     high=priors_change_range,
    #     shape=(priors_size,),
    # )

    param_space["mu"] = gym.spaces.Box(
        low=-mu_change_range,
        high=mu_change_range,
        shape=(means_size,),
    )
    return gym.spaces.Dict(param_space)


def get_meta_ac_space(cfg):
    """Returns high level agent's action space as gym.spaces.Dict
    Note that the high level agent must predict:
    - One hot vector of skill_dim size for sequencing
    - GMM change to refine gmms
    """
    parameter_space = get_ref_param_space(cfg)
    mu_high = np.ones(parameter_space["mu"].shape[0])
    if "priors" in parameter_space:
        priors_high = np.ones(parameter_space["priors"].shape[0])
        action_high = np.concatenate((priors_high, mu_high), axis=-1)
    else:
        action_high = mu_high
    action_low = -action_high
    action_space = gym.spaces.Box(action_low, action_high)
    return action_space


def get_refine_dict(cfg, gmm_change):
    """Takes refine vector and returns a dictionary for easy gmm update"""
    ref_param_space = get_ref_param_space(cfg)
    if "priors" in ref_param_space:
        size_priors = ref_param_space["priors"].shape[0]
        size_mu = ref_param_space["mu"].shape[0]

        priors = gmm_change[:size_priors] * ref_param_space["priors"].high
        mu = (
            gmm_change[size_priors : size_priors + size_mu] * ref_param_space["mu"].high
        )

        refine_dict = {"mu": mu, "priors": priors}
    else:
        mu = gmm_change * ref_param_space["mu"].high
        refine_dict = {"mu": mu}
    return refine_dict

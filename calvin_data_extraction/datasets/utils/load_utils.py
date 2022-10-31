import hydra
import numpy as np
from typing import List, Dict
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from pathlib import Path


def get_transforms(transforms_list: List[dict] = None):
    map_interp_mode = {
        "bilinear": InterpolationMode.BILINEAR,
        "nearest": InterpolationMode.NEAREST,
    }
    inst_transf_list = []
    for transform in transforms_list:
        if "interpolation" in transform:
            transform = OmegaConf.to_container(transform)
            transform["interpolation"] = map_interp_mode[transform["interpolation"]]
        inst_transf_list.append(hydra.utils.instantiate(transform))
    return transforms.Compose(inst_transf_list)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())

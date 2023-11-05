import torch
import torch.nn as nn
import numpy as np

from pl_bolts.models import AE
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)  # , resnet50, ResNet50_Weights
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


def calc_out_size(w, h, kernel_size, padding=0, stride=1):
    width = (w - kernel_size + 2 * padding) // stride + 1
    height = (h - kernel_size + 2 * padding) // stride + 1
    return width, height


class PTEncoder(nn.Module):
    def __init__(self, model_name):
        super(PTEncoder, self).__init__()
        self.no_encoder = False
        if model_name == "cifar10-resnet18":
            self.wrap = AE(input_height=32).from_pretrained("cifar10-resnet18").encoder
            self.mean = np.array([x / 255.0 for x in [125.3, 123.0, 113.9]])
            self.std = np.array([x / 255.0 for x in [63.0, 62.1, 66.7]])
            self.feature_size = 512
            self.wrap.eval()
        elif model_name == "imagenet-resnet18":
            self.wrap = torch.nn.Sequential(
                *(list(resnet18(weights=ResNet18_Weights.DEFAULT).children())[:-2])
            )
            preprocess = ResNet18_Weights.DEFAULT.transforms()
            self.mean = np.array(preprocess.mean)
            self.std = np.array(preprocess.std)
            self.feature_size = 512
            self.wrap.eval()
        else:
            self.no_encoder = True
            # self.feature_size = 21
            self.feature_size = 25

    def preprocess(self, x, mean, std):
        norm = T.Compose(
            [
                T.Resize(32, interpolation=InterpolationMode.BILINEAR),
                T.Normalize(mean=mean, std=std),
            ]
        )
        o = norm(x / 255.0)
        return o

    def forward(self, x):
        with torch.no_grad():
            if self.no_encoder:
                return None
            o = self.wrap(self.preprocess(x, self.mean, self.std))
        return o

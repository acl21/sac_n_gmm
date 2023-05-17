import torch
import numpy as np
from torchvision import transforms


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.std = torch.tensor(std)
        self.mean = torch.tensor(mean)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert isinstance(tensor, torch.Tensor)
        device = tensor.device
        if device != self.std.device:
            self.std = self.std.to(device)
        if device != self.mean.device:
            self.mean = self.mean.to(device)
        return tensor + torch.randn(tensor.size(), device=device) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class ArrayToTensor(object):
    """Transforms np array to tensor."""

    def __call__(self, array: np.ndarray, device: torch.device = "cpu") -> torch.Tensor:
        assert isinstance(array, np.ndarray)
        return torch.from_numpy(array).to(device)


class PreprocessImage(object):
    """Preprocesses np image to tensor image."""

    def __call__(self, array: np.ndarray) -> torch.Tensor:
        assert isinstance(array, np.ndarray)
        array = np.transpose(array, (2, 0, 1))
        tensor = torch.from_numpy(array).type(torch.FloatTensor)
        mean, std = tensor.mean([1, 2]), tensor.std([1, 2])
        tensor = transforms.Normalize(mean, std)(tensor)
        tensor = transforms.Resize(64)(tensor)
        tensor = transforms.Grayscale(1)(tensor)
        tensor = tensor.unsqueeze(0)
        return tensor

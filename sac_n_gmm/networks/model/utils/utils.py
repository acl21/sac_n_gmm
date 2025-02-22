import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

activation_map = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
}


def get_activation(activation):
    if activation is None:
        activation = nn.Identity()
    elif isinstance(activation, str):
        activation = activation_map[activation.lower()]()
    return activation


def rmap(func, x):
    """Recursively applies `func` to list or dictionary `x`."""
    if isinstance(x, dict):
        return OrderedDict([(k, rmap(func, v)) for k, v in x.items()])
    if isinstance(x, list):
        return [rmap(func, v) for v in x]
    return func(x)


# from https://github.com/denisyarats/drq/blob/master/utils.py#L62
def weight_init(tensor):
    if isinstance(tensor, nn.Linear):
        nn.init.orthogonal_(tensor.weight.data)
        if tensor.bias is not None:
            tensor.bias.data.fill_(0.0)
    elif isinstance(tensor, nn.Conv2d) or isinstance(tensor, nn.ConvTranspose2d):
        tensor.weight.data.fill_(0.0)
        if tensor.bias is not None:
            tensor.bias.data.fill_(0.0)
        mid = tensor.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        # nn.init.orthogonal_(tensor.weight.data[:, :, mid, mid], gain)
        nn.init.orthogonal_(tensor.weight.data, gain)


def weight_init_xavier_uniform(tensor):
    if isinstance(tensor, nn.Linear):
        nn.init.xavier_uniform_(tensor.weight)
        tensor.bias.data.fill_(0.0)
    elif isinstance(tensor, nn.Conv2d) or isinstance(tensor, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(tensor.weight, gain=nn.init.calculate_gain("relu"))
        tensor.bias.data.fill_(0.0)


def weight_init_small(tensor):
    if isinstance(tensor, nn.Linear):
        nn.init.orthogonal_(tensor.weight.data, gain=0.01)
        if tensor.bias is not None:
            tensor.bias.data.fill_(0.0)


def weight_init_xavier_normal(tensor):
    if isinstance(tensor, nn.Linear):
        nn.init.xavier_normal(tensor.weight.data)
        if tensor.bias is not None:
            tensor.bias.data.fill_(0.0)


class CNN(nn.Module):
    def __init__(self, cfg, input_dim):
        super().__init__()

        self.convs = nn.ModuleList()
        d_prev = input_dim
        d = cfg.encoder_conv_dim
        h, w = cfg.encoder_image_size
        for k, s in zip(cfg.encoder_kernel_size, cfg.encoder_stride):
            self.convs.append(nn.Conv2d(d_prev, d, int(k), int(s)))
            w = int(np.floor((w - (int(k) - 1) - 1) / int(s) + 1))
            d_prev = d

        print(f"Output of CNN ({w*w*d}) = {w} x {w} x {d}")
        self.output_dim = cfg.encoder_conv_output_dim

        self.fc = nn.Linear(w * w * d, self.output_dim)
        self.ln = nn.LayerNorm(self.output_dim)

        self.apply(weight_init)

    def forward(self, ob, detach_conv=False):
        out = ob
        for conv in self.convs:
            out = F.relu(conv(out))
        out = out.flatten(start_dim=1)

        if detach_conv:
            out = out.detach()

        out = self.fc(out)
        out = self.ln(out)
        out = torch.tanh(out)

        return out

    # from https://github.com/MishaLaskin/rad/blob/master/encoder.py
    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i, conv in enumerate(self.convs):
            assert type(source.convs[i]) == type(conv)
            conv.weight = source.convs[i].weight
            conv.bias = source.convs[i].bias


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims,
        activation,
        norm=False,
        out_act=False,
        small_weight=False,
    ):
        super().__init__()
        activation = get_activation(activation)
        dims = [input_dim] + list(hidden_dims) + [output_dim]

        fcs = []
        for prev_d, d in zip(dims[:-1], dims[1:]):
            fcs.append(nn.Linear(prev_d, d, bias=not norm))
            if norm:
                fcs.append(nn.LayerNorm(d))
            fcs.append(activation)

        if not out_act:
            fcs = fcs[: -3 if norm else -2]
            fcs.append(nn.Linear(prev_d, d))
        self.fcs = nn.Sequential(*fcs)
        self.output_dim = output_dim

        self.apply(weight_init)
        if small_weight:
            self.apply(weight_init_small)

    def forward(self, ob):
        return self.fcs(ob)


def flatten_ob(ob: dict, ac=None):
    """
    Flattens the observation dictionary. The observation dictionary
    can either contain a single ob, or a batch of obs.
    Any images must be flattened to 1D tensors, but
    we must be careful to check if we are doing a single instance
    or batch before we flatten.

    Returns a list of dim [N x D] where N is batch size and D is sum of flattened
    dims of observations
    """
    inp = []
    images = []
    single_ob = False
    for k, v in ob.items():
        if k in ["camera_ob", "depth_ob", "segmentation_ob"]:
            images.append(v)
        else:
            if len(v.shape) == 1:
                single_ob = True
            inp.append(v)
    # concatenate images into 1D
    for image in images:
        if single_ob:
            img = torch.flatten(image)
        else:  # batch of obs, flatten after bs dim
            img = torch.flatten(image, start_dim=1)
        inp.append(img)
    # now flatten into Nx1 tensors
    if single_ob:
        inp = [x.unsqueeze(0) for x in inp]

    if ac is not None:
        ac = list(ac.values())
        if len(ac[0].shape) == 1:
            ac = [x.unsqueeze(0) for x in ac]
        inp.extend(ac)
    inp = torch.cat(inp, dim=-1)
    return inp


def flatten_ac(ac: dict):
    ac = list(ac.values())
    if len(ac[0].shape) == 1:
        ac = [x.unsqueeze(0) for x in ac]
    ac = torch.cat(ac, dim=-1)
    return ac

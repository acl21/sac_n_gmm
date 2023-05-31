import torch
import torch.nn as nn
import torch.nn.functional as F
from examples.CALVINExperiment.rl.networks.critics.critic import Critic


class DenseNetCritic(Critic):
    def __init__(
        self,
        input_dim: int,
        hidden_depth: int = 2,
        hidden_dim: int = 256,
    ):
        super(DenseNetCritic, self).__init__(
            input_dim=input_dim,
            hidden_depth=hidden_depth,
            hidden_dim=hidden_dim,
        )
        self.Q1 = DenseNetQNetwork(
            input_dim=input_dim, hidden_depth=hidden_depth, hidden_dim=hidden_dim
        )
        self.Q2 = DenseNetQNetwork(
            input_dim=input_dim, hidden_depth=hidden_depth, hidden_dim=hidden_dim
        )


class DenseNetQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_depth: int = 2,
        hidden_dim: int = 256,
    ):
        super(DenseNetQNetwork, self).__init__()
        fc_in_features = input_dim
        self.fc_layers = []
        for _ in range(hidden_depth):
            self.fc_layers.append(nn.Linear(fc_in_features, hidden_dim))
            fc_in_features += hidden_dim
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.out = nn.Linear(fc_in_features, 1)

    def forward(self, state, action):
        fc_input = torch.cat((state, action), dim=-1)
        num_layers = len(self.fc_layers)
        for i in range(num_layers):
            fc_output = F.silu(self.fc_layers[i](fc_input))
            fc_input = torch.cat([fc_input, fc_output], dim=-1)
        return self.out(fc_input)

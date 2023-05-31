import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_depth: int = 2,
        hidden_dim: int = 256,
    ):
        super(Critic, self).__init__()
        self.Q1 = QNetwork(
            input_dim=input_dim, hidden_depth=hidden_depth, hidden_dim=hidden_dim
        )
        self.Q2 = QNetwork(
            input_dim=input_dim, hidden_depth=hidden_depth, hidden_dim=hidden_dim
        )

    def forward(self, observation, action):
        state = observation
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_depth: int = 2, hidden_dim: int = 256):
        super(QNetwork, self).__init__()
        self.fc_layers = [nn.Linear(input_dim, hidden_dim)]
        for i in range(hidden_depth - 1):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        fc_input = torch.cat((state, action), dim=-1)
        num_layers = len(self.fc_layers)
        x = F.silu(self.fc_layers[0](fc_input))
        for i in range(1, num_layers):
            x = F.silu(self.fc_layers[i](x))
        return self.out(x)

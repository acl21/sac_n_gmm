import torch
import torch.nn as nn
import torch.nn.functional as F
from examples.CALVINExperiment.rl.networks.actors.actor import Actor

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0


class DenseNetActor(Actor):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_depth: int = 2,
        hidden_dim: int = 256,
        init_w: float = 1e-3,
    ):
        super(DenseNetActor, self).__init__(
            input_dim=input_dim,
            action_dim=action_dim,
            hidden_depth=hidden_depth,
            hidden_dim=hidden_dim,
            init_w=init_w,
        )
        del self.fc_layers
        self.fc_layers = []
        fc_in_features = input_dim
        for _ in range(hidden_depth):
            self.fc_layers.append(nn.Linear(fc_in_features, hidden_dim))
            fc_in_features += hidden_dim
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.fc_mean = nn.Linear(fc_in_features, action_dim)
        self.fc_log_std = nn.Linear(fc_in_features, action_dim)
        self.fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc_log_std.bias.data.uniform_(-init_w, init_w)
        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.uniform_(-init_w, init_w)

    def get_last_hidden_state(self, fc_input):
        num_layers = len(self.fc_layers)
        for i in range(num_layers):
            fc_output = F.silu(self.fc_layers[i](fc_input))
            fc_input = torch.cat([fc_input, fc_output], dim=-1)
        return fc_input

    def forward(self, observation):
        state = observation
        x = self.get_last_hidden_state(state)
        mean = self.fc_mean(x)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()
        return mean, std

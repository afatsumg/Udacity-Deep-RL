# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)

class Actor(nn.Module):
    """Tanh-squashed Gaussian policy for a single agent."""
    def __init__(self, state_size, action_size, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, action_size)
        self.log_std = nn.Linear(hidden, action_size)
        self.apply(init_weights)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-20, 2)  # stable range
        return mu, log_std

    def sample(self, state):
        """Returns (action, log_prob, pre_tanh_value)."""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()  # reparameterization
        a = torch.tanh(x)
        # log_prob with tanh correction
        log_prob = dist.log_prob(x) - torch.log(1 - a.pow(2) + EPS)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return a, log_prob, mu

class TwinCritic(nn.Module):
    """Twin Q-networks (centralized) taking full_state and full_action."""
    def __init__(self, full_state_size, full_action_size, hidden=256):
        super().__init__()
        in_dim = full_state_size + full_action_size

        # Q1
        self.q1_fc1 = nn.Linear(in_dim, hidden)
        self.q1_fc2 = nn.Linear(hidden, hidden)
        self.q1_out = nn.Linear(hidden, 1)

        # Q2
        self.q2_fc1 = nn.Linear(in_dim, hidden)
        self.q2_fc2 = nn.Linear(hidden, hidden)
        self.q2_out = nn.Linear(hidden, 1)

        self.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        return q1, q2

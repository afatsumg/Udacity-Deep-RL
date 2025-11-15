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
    """Tanh-squashed Gaussian policy for a single agent.
    
    This actor outputs mu and log_std for a Gaussian policy. 
    Reparameterization trick is used for differentiable sampling.
    Tanh squashing ensures actions remain in [-1, 1].
    """
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
        x = dist.rsample()  # reparameterization trick
        a = torch.tanh(x)
        # log_prob with tanh correction to account for squashing
        log_prob = dist.log_prob(x) - torch.log(1 - a.pow(2) + EPS)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return a, log_prob, mu


class TwinCritic(nn.Module):
    """Twin Q-networks (centralized) taking full_state and full_action.
    
    In MASAC, we use two critics for the same input (s, a) to reduce overestimation bias:
        Q1(s,a), Q2(s,a)
    During learning, the target Q is computed as min(Q1_target, Q2_target), which improves stability.
    
    Inputs:
        state: concatenated states of all agents [B, N*S]
        action: concatenated actions of all agents [B, N*A]
    Outputs:
        q1, q2: twin Q-values for the given joint state-action
    """
    def __init__(self, full_state_size, full_action_size, hidden=256):
        super().__init__()
        in_dim = full_state_size + full_action_size

        # Q1 network
        self.q1_fc1 = nn.Linear(in_dim, hidden)
        self.q1_fc2 = nn.Linear(hidden, hidden)
        self.q1_out = nn.Linear(hidden, 1)

        # Q2 network
        self.q2_fc1 = nn.Linear(in_dim, hidden)
        self.q2_fc2 = nn.Linear(hidden, hidden)
        self.q2_out = nn.Linear(hidden, 1)

        self.apply(init_weights)

    def forward(self, state, action):
        # concatenate states and actions to form joint input
        x = torch.cat([state, action], dim=1)

        # forward pass through Q1
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        # forward pass through Q2
        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)

        # return twin Q-values
        return q1, q2

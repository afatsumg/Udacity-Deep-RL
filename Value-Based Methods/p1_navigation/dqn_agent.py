import numpy as np
import random
from collections import namedtuple, deque


import torch
import torch.nn.functional as F
import torch.optim as optim

from model import DuelingQNetwork

BUFFER_SIZE = 500000    # replay buffer size
BATCH_SIZE = 64         # batch size for sampling from replay buffer
GAMMA = 0.99            # discount rate
TAU = 5e-3              # for soft update of target parameters
LR = 1e-4               # learning rate
UPDATE_EVERY = 4  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory with N-step support 
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, n_step=3, gamma=GAMMA)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        
        best_local_actions = self.qnetwork_local(states).max(1)[1].unsqueeze(1)
        double_dqn_targets = self.qnetwork_target(next_states)
        Q_targets_next = torch.gather(double_dqn_targets, 1, best_local_actions)

        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples (with N-step support)."""

    def __init__(self, action_size, buffer_size, batch_size, seed, n_step=3, gamma=0.99):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.n_step_buffer = deque(maxlen=n_step)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.n_step = n_step
        self.gamma = gamma

    def add(self, state, action, reward, next_state, done):
        """Add a new experience and handle N-step transitions."""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Only add to main memory when buffer full or done
        if len(self.n_step_buffer) == self.n_step or done:
            R, next_s, d = self._get_n_step_info()
            s, a, *_ = self.n_step_buffer[0]
            e = self.experience(s, a, R, next_s, d)
            self.memory.append(e)

    def _get_n_step_info(self):
        """Compute N-step discounted reward and next state."""
        R, next_s, d = 0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
        for idx, (_, _, r, _, done) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * r
            if done:
                break
        return R, next_s, d

    def sample(self):
        """Sample a batch of experiences."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

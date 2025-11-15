# agent.py
import numpy as np
import random
from collections import deque, namedtuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, TwinCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, full_state_dim, full_action_dim, buffer_size=int(1e6), batch_size=256, seed=0):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", ["states", "actions", "rewards", "next_states", "dones"])
        random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        """states/actions are flattened joint arrays: shape (full_dim,)"""
        self.memory.append(self.experience(states, actions, rewards, next_states, dones))

    def sample(self):
        batch = random.sample(self.memory, k=self.batch_size)
        states = torch.FloatTensor(np.vstack([e.states for e in batch])).to(device)
        actions = torch.FloatTensor(np.vstack([e.actions for e in batch])).to(device)
        rewards = torch.FloatTensor(np.vstack([e.rewards for e in batch])).to(device)  # shape [B, N]
        next_states = torch.FloatTensor(np.vstack([e.next_states for e in batch])).to(device)
        dones = torch.FloatTensor(np.vstack([e.dones for e in batch])).to(device)        # shape [B, N]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class MASAC:
    def __init__(
        self,
        num_agents,
        state_size,
        action_size,
        lr_actor=3e-4,
        lr_critic=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.1,
        hidden_actor=128,
        hidden_critic=256,
        batch_size=256,
        buffer_size=int(1e6),
        update_steps=2,
        seed=0
    ):
        random.seed(seed)

        self.N = num_agents
        self.S = state_size
        self.A = action_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.update_steps = update_steps

        self.full_state = self.N * self.S
        self.full_action = self.N * self.A

        # actors per agent
        self.actors = [Actor(self.S, self.A, hidden=hidden_actor).to(device) for _ in range(self.N)]
        self.actors_opt = [optim.Adam(a.parameters(), lr=lr_actor) for a in self.actors]

        # centralized twin critic + target
        self.critic = TwinCritic(self.full_state, self.full_action, hidden=hidden_critic).to(device)
        self.critic_target = TwinCritic(self.full_state, self.full_action, hidden=hidden_critic).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # replay
        self.replay = ReplayBuffer(self.full_state, self.full_action, buffer_size=buffer_size, batch_size=batch_size, seed=seed)

    def act(self, states, deterministic=False):
        """
        states: numpy array shape (N, S)
        returns actions numpy array shape (N, A)
        """
        acts = []
        for i in range(self.N):
            s = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
            if deterministic:
                mu, _ = self.actors[i].forward(s)
                a = torch.tanh(mu).cpu().data.numpy().squeeze()
            else:
                a, _, _ = self.actors[i].sample(s)
                a = a.cpu().data.numpy().squeeze()
            acts.append(a)
        return np.vstack(acts)

    def step(self, states, actions, rewards, next_states, dones):
        """
        states: np array shape (N, S)
        actions: np array shape (N, A)
        rewards: np array shape (N,) or (N,1)
        next_states: np array shape (N, S)
        dones: np array shape (N,) or (N,1)

        We store flattened joint representation for centralized critic.
        """
        s_flat = np.reshape(states, -1)
        a_flat = np.reshape(actions, -1)
        ns_flat = np.reshape(next_states, -1)
        r_flat = np.reshape(rewards, -1)   # shape (N,)
        d_flat = np.reshape(dones, -1)
        # store rewards & dones as arrays per agent (we keep shape N to compute per-agent done)
        self.replay.add(s_flat, a_flat, r_flat, ns_flat, d_flat)

        # update
        if len(self.replay) > self.replay.batch_size:
            for _ in range(self.update_steps):
                self.learn()

    def learn(self):
        states_b, actions_b, rewards_b, next_states_b, dones_b = self.replay.sample()
        # states_b: [B, full_state], actions_b: [B, full_action]
        # rewards_b: [B, N], dones_b: [B, N]

        # -------------------------------
        # 1) Compute next actions & log_probs (for all agents)
        # For agent i:
        # a'_i ~ pi_phi(a_i | s'_i)
        # Form full next action vector: a' = [a'_1, ..., a'_N]
        # -------------------------------
        next_actions = []
        next_logp = 0.0
        for i in range(self.N):
            s_i = next_states_b[:, i*self.S:(i+1)*self.S]
            a_i, logp_i, _ = self.actors[i].sample(s_i)
            next_actions.append(a_i)
            next_logp = next_logp + logp_i  # sum log probs over all agents (entropy term)
        next_actions_cat = torch.cat(next_actions, dim=1)  # [B, full_action]

        # -------------------------------
        # 2) Compute target Q values using Twin-Q networks
        # Q_min(s', a') = min(Q1_target(s', a'), Q2_target(s', a')) - alpha * sum_i log pi(a'_i|s'_i)
        # Convert rewards to cooperative scalar target: r_sum = sum_i r_i
        # y = r_sum + gamma * (1 - done_any) * Q_min(s', a')
        # -------------------------------
        with torch.no_grad():
            q1_next, q2_next = self.critic_target(next_states_b, next_actions_cat)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_logp
            r_sum = rewards_b.sum(dim=1, keepdim=True)
            d_any = dones_b.max(dim=1, keepdim=True)[0]
            y = r_sum + (1.0 - d_any) * self.gamma * q_next

        # -------------------------------
        # 3) Critic update
        # Loss: L_critic = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
        # Update critic network parameters
        # -------------------------------
        q1, q2 = self.critic(states_b, actions_b)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # -------------------------------
        # 4) Actor updates (per agent)
        # Maximize expected Q - alpha * log_prob:
        # L_actor_i = E[alpha * log pi(a_i|s_i) - Q_min(s, a_pred)]
        # a_pred = [a_1_pred, ..., a_i, ..., a_N_pred] 
        # with other agents' actions detached (no gradient through them)
        # -------------------------------
        for i in range(self.N):
            s_i = states_b[:, i*self.S:(i+1)*self.S]
            a_i, logp_i, _ = self.actors[i].sample(s_i)

            actions_pred = []
            for j in range(self.N):
                if j == i:
                    actions_pred.append(a_i)
                else:
                    s_j = states_b[:, j*self.S:(j+1)*self.S]
                    with torch.no_grad():
                        a_j, _, _ = self.actors[j].sample(s_j)
                    actions_pred.append(a_j)
            actions_pred_cat = torch.cat(actions_pred, dim=1)

            q1_pi, q2_pi = self.critic(states_b, actions_pred_cat)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha * logp_i - q_pi).mean()

            self.actors_opt[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0)
            self.actors_opt[i].step()

        # -------------------------------
        # 5) Soft update of target critic network
        # theta_target = tau * theta + (1 - tau) * theta_target
        # -------------------------------
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


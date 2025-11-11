import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PPOAgent:
    def __init__(self, model, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2, value_coef=0.5, ent_coef=0.01):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.ent_coef = ent_coef

    def compute_gae(self, rewards, masks, values):
        """
        rewards, masks, values: list of tensors, her tensor shape = (num_agents,)
        döndürür: list of tensors, her tensor shape = (num_agents,)
        """
        rewards = torch.stack(rewards)              # (T, N)
        masks = torch.stack(masks)                  # (T, N)
        values = torch.stack(values).squeeze(-1)   # (T, N)

        T, N = rewards.shape
        returns = torch.zeros_like(rewards)
        gae = torch.zeros(N)
        next_value = torch.zeros(N)

        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_value * masks[t] - values[t]
            gae = delta + self.gamma * self.lam * masks[t] * gae
            returns[t] = gae + values[t]
            next_value = values[t]

        return [r for r in returns]  # liste içinde (N,) tensorler

    def ppo_update(self, states, actions, log_probs_old, returns, values, epochs=10, batch_size=64):
        """
        states, actions, log_probs_old, values: list of tensors, her tensor shape = (num_agents, dim)
        returns: list of tensors, her tensor shape = (num_agents,)
        """
        # --- Flatten T × N → batch ---
        states = torch.cat(states, dim=0)          # (T*N, state_dim)
        actions = torch.cat(actions, dim=0)        # (T*N, action_dim)
        log_probs_old = torch.cat(log_probs_old, dim=0).detach()  # (T*N,)
        values = torch.cat(values, dim=0).detach().squeeze(-1)   # (T*N,)
        returns = torch.cat(returns, dim=0).detach()             # (T*N,)

        for _ in range(epochs):
            for i in range(0, len(states), batch_size):
                s_batch = states[i:i+batch_size]
                a_batch = actions[i:i+batch_size]
                old_log_prob_batch = log_probs_old[i:i+batch_size]
                return_batch = returns[i:i+batch_size]
                value_batch = values[i:i+batch_size]

                # --- Model evaluate ---
                log_prob, entropy, value = self.model.evaluate(s_batch, a_batch)
                value = value.squeeze(-1)  # (batch_size,)

                # --- Advantage normalize ---
                advantage = return_batch - value_batch
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # --- PPO loss ---
                ratio = (log_prob - old_log_prob_batch).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(value, return_batch)  # target detached
                loss = actor_loss + self.value_coef * critic_loss - self.ent_coef * entropy.mean()

                # --- Backprop ---
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

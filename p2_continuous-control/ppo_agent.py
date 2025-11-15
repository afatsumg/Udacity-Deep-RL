import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class PPOAgent:
    def __init__(self, model, lr=3e-4, gamma=0.99, lam=0.95, eps_clip=0.2, value_coef=0.5, ent_coef=0.01):
        # PPO policy/value network
        self.model = model

        # Adam optimizer used for joint actor–critic update
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Discount factor γ
        self.gamma = gamma

        # GAE smoothing factor λ
        self.lam = lam

        # PPO clipping parameter ε
        self.eps_clip = eps_clip

        # Critic loss coefficient (value loss weight)
        self.value_coef = value_coef

        # Entropy bonus coefficient (promotes exploration)
        self.ent_coef = ent_coef

    def compute_gae(self, rewards, masks, values):
        """
        GAE(λ) calculation:
        δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        A_t = δ_t + γλ A_{t+1}

        returns = A_t + V(s_t)
        """
        # Stack to (T, N)
        rewards = torch.stack(rewards)
        masks = torch.stack(masks)
        values = torch.stack(values).squeeze(-1)

        T, N = rewards.shape

        # Output buffers
        returns = torch.zeros_like(rewards)
        gae = torch.zeros(N)
        next_value = torch.zeros(N)

        # --- GAE backward recursion ---
        for t in reversed(range(T)):
            # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * masks[t] - values[t]

            # A_t = δ_t + γλ A_{t+1}
            gae = delta + self.gamma * self.lam * masks[t] * gae

            # Return_t = A_t + V(s_t)
            returns[t] = gae + values[t]

            # Move to previous timestep
            next_value = values[t]

        return [r for r in returns]

    def ppo_update(self, states, actions, log_probs_old, returns, values, epochs=10, batch_size=64):
        """
        PPO objective:
        L_clip = min(
            r(θ) * A,
            clip(r(θ), 1 - ε, 1 + ε) * A
        )
        where r(θ) = π_θ(a|s) / π_old(a|s)
        """

        # Flatten T × N → (TN, dim)
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)

        # Detached old log probabilities (π_old(a|s))
        log_probs_old = torch.cat(log_probs_old, dim=0).detach()

        # Baseline V(s)
        values = torch.cat(values, dim=0).detach().squeeze(-1)

        # GAE returns
        returns = torch.cat(returns, dim=0).detach()

        # --- Multi-epoch PPO updates ---
        for _ in range(epochs):
            # Mini-batch SGD
            for i in range(0, len(states), batch_size):

                # Select mini-batch
                s_batch = states[i:i+batch_size]
                a_batch = actions[i:i+batch_size]
                old_log_prob_batch = log_probs_old[i:i+batch_size]
                return_batch = returns[i:i+batch_size]
                value_batch = values[i:i+batch_size]

                # --- Evaluate πθ(a|s), entropy and Vθ(s) ---
                # log_prob = log πθ(a|s)
                # entropy = entropy(πθ)
                # value = Vθ(s)
                log_prob, entropy, value = self.model.evaluate(s_batch, a_batch)
                value = value.squeeze(-1)

                # --- Compute advantage ---
                # A = returns - V_old(s)
                advantage = return_batch - value_batch

                # Normalize advantage (critical for PPO stability)
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                # --- Probability ratio r(θ) = exp(log πθ - log π_old) ---
                ratio = (log_prob - old_log_prob_batch).exp()

                # --- Clipped surrogate objective ---
                # surr1 = r(θ) * A
                surr1 = ratio * advantage

                # surr2 = clip(r(θ)) * A
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.eps_clip,
                    1.0 + self.eps_clip
                ) * advantage

                # Actor loss: negative clipped objective
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss: MSE(V(s), returns)
                critic_loss = nn.MSELoss()(value, return_batch)

                # Total PPO loss:
                # L = L_actor + c1 * L_critic - c2 * entropy
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.ent_coef * entropy.mean()
                )

                # --- Backprop ---
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

# Reacher Multi-Agent Environment Training Report (PPO)

## 1. Project Goal

The primary objective was to train a Proximal Policy Optimization (PPO) agent to solve the **Unity Reacher Multi-Agent Environment**. The environment consists of 20 distinct agents, each controlling a double-jointed arm. The task is to track a moving target for maximum reward.

The solution requirement, as specified by the environment, is achieving an **average score of +30.0** over 100 consecutive episodes.

---

## 2. Environment and Agent Setup

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Algorithm** | PPO (Multi-Agent) | Centralized training, decentralized execution. |
| **State Dimension** | 33 | Continuous (Position, Rotation, Velocity) |
| **Action Dimension** | 4 | Continuous (Torques for two joints) |
| **Number of Agents** | 20 | Experience collected in parallel. |
| **Discount Factor ($\gamma$)** | 0.99 | Long-term view on rewards. |
| **Value Coefficient** | 1.0 | Prioritized stability of the Critic Network. |
| **Clipping Parameter ($\epsilon$)** | 0.2 | Standard PPO clipping range. |

---

## 3. Training Process and Hyperparameter Tuning

The training process began with conservative hyperparameters to ensure stability, but quickly hit a significant **plateau** around an average score of **23.40**. The subsequent tuning aimed to break this plateau without causing catastrophic policy failure (policy collapse).

### Phase 1: Initial Stability and Plateau (Episode 1 – 242)

* **Initial LR:** $1.0 \times 10^{-4}$ (Linear decay via `LambdaLR`).
* **Initial Ent Coef:** $0.01$.
* **Result:** The agent rapidly climbed to an average score of **23.40** (around Episode 242) but then stalled. The continuous reduction of the Learning Rate via `LambdaLR` led to insufficient momentum to explore policies beyond the local optimum.

### Phase 2: Attempts to Break the Plateau (Failure Modes)

| Attempt | LR Setting | Ent Coef | $\lambda$ (GAE) | Result | Reason for Failure |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2.1** | $1.5 \times 10^{-4}$ (Fixed) | $0.03$ (High) | $0.95$ | **Collapse** ($23.40 \to 21.55$) | High LR and high exploration caused the agent to quickly unlearn its stable $23.40$ policy. |
| **2.2** | $1.5 \times 10^{-4}$ (Fixed) | $0.005$ (Low) | $0.95$ | **Collapse** ($23.40 \to 21.57$) | Even with minimal exploration, the aggressive LR was too large, causing the policy update steps to overshoot optimal weights. |

### Phase 3: The Final Solution (Return to Stability + Momentum)

Based on the failure analysis, the key was to combine the **stable LR** with a minor adjustment to the **GAE ($\lambda$) factor** to focus the agent's learning. The solution used the best saved weights (at score 23.40) and modified the underlying PPO mechanics for fine-tuning.

| Final Setting | Value | Rationale |
| :--- | :--- | :--- |
| **Learning Rate (LR)** | **$1.0 \times 10^{-4}$** (Fixed) | Returned to the highest stable LR. The `LambdaLR` scheduler was removed to maintain constant momentum. |
| **Ent Coef** | **$0.01$** (Medium) | Restored to the level that initially proved successful for moderate exploration. |
| **GAE Lambda ($\lambda$)** | **$0.80$** (Reduced) | This was the key stabilization measure. Reducing $\lambda$ from $0.95$ to $0.80$ forced the agent to consider a **shorter time horizon** when calculating advantages. This results in **safer, less volatile** policy updates, preventing the agent from relying too heavily on distant, uncertain rewards. |

---

## 4. Results and Conclusion

The adjustments in Phase 3 immediately broke the plateau and initiated a swift, stable climb towards the target score.

| Metric | Score @ Episode 242 | Score @ Episode 310 |
| :--- | :--- | :--- |
| **Average Score (100-ep)** | 23.40 | **30.01** |
| **Training Time** | Plateau | Swift 68 episodes climb |

The environment was successfully solved in **Episode 310** when the 100-episode moving average reached **30.01**.

The final learning curve demonstrates the successful breakthrough of the plateau and the subsequent stable convergence:



**Conclusion:** For high-dimensional, multi-agent continuous control problems like Reacher, aggressive learning rates can easily destabilize the policy, even with high Entropies designed to promote robust exploration. The final solution hinged on prioritizing **policy stability** by maintaining a safe LR and reducing the Generalised Advantage Estimation (GAE) horizon ($\lambda$), allowing the agent to perform safe, yet persistent, fine-tuning.
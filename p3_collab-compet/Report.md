# **MASAC Training Report – Unity Tennis Environment**

## **1. Overview**

This report documents the training of a Multi-Agent Soft Actor-Critic (MASAC) agent on the **Unity ML-Agents Tennis** environment using an optimized implementation of MASAC with tuned hyperparameters, prioritized updates, and robust checkpointing.

The final model successfully solved the environment in:

### 🎉 **2749 episodes — 0.501 average score over last 100 episodes**

----------

## **2. Environment Details**

**Unity Environment:** Tennis  
**Number of Agents:** 2  
**State Size:** 24 (3 stacked frames of 8 features)  
**Action Size:** 2  
**Action Type:** Continuous (player movement & jump direction)

The goal is to keep the ball in play and outplay the opponent. Agents are rewarded based on rally length and penalized for letting the ball hit the ground.

----------

## **3. MASAC Implementation Summary**
### **MASAC Algorithm Steps**

1. Collect transitions from all agents:
   - states (N × S)
   - actions (N × A)
   - rewards (N)
   - next_states (N × S)
   - dones (N)

2. Store transition in replay buffer.

3. Sample random batch from buffer.

4. Compute target actions using target actors.

5. Compute target Q-value using target critic.

6. Update critic networks via MSE loss.

7. Update actor networks using entropy-regularized objective.

8. Soft-update target networks:
   θ' ← τθ + (1−τ)θ'

9. Repeat for all steps across all episodes.

The agent uses:

-   **Centralized Critic** (shared across agents)
    
-   **Decentralized Actors** (one per agent)
    
-   **Entropy-regularized learning** for stability
    
-   **Soft target updates**
    
-   **Replay buffer with batches of 256**
    
-   **2 gradient updates per environment step**

### **Hyperparameters**

| Parameter | Value |
|----------|--------|
| Learning Rate (Actor) | 3e-4 |
| Learning Rate (Critic) | 3e-4 |
| Gamma (Discount) | 0.99 |
| Soft Update Tau | 0.005 |
| Entropy Coefficient α | 0.1 |
| Batch Size | 256 |
| Replay Buffer | 1e6 |
| Update Steps per Action | 2 |
| Hidden Actor Layers | 128 |
| Hidden Critic Layers | 256 |
| Seed | 0 |

### **Training Progress**

| Episode | Avg Score (last 100) |
|---------|------------------------|
| 100 | 0.012 |
| 500 | 0.029 |
| 900 | 0.044 |
| 1400 | 0.072 |
| 1800 | 0.089 |
| 2000 | 0.137 |
| 2300 | 0.180 |
| 2500 | 0.258 |
| 2600 | 0.308 |
| 2700 | 0.364 |
| 2749 | **0.501 (SOLVED)** |

## ** Recommendations for Future Improvement**

If further optimization is desired:

### **1. Automatic Entropy Tuning**

Let the algorithm learn α dynamically.

### **2. Prioritized Experience Replay**

Boosts learning speed ~20–30%.

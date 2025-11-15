
## 📚 **Overview**

This repository contains my solutions to the **Udacity Deep Reinforcement Learning Nanodegree**, including implementations of modern RL algorithms across value-based, continuous-control, and multi-agent environments.

Projects:

1.  **Value-Based Methods** (DQN, Double DQN, Dueling DQN, PER)
    
2.  **Continuous Control — PPO solution for 20-agent Reacher**
    
3.  **Collaboration and Competition — MASAC solution for Tennis**
    

----------

# 🧩 **Project 1 – Value-Based Methods**
### **🎮 Environment: Banana Collector (Unity ML-Agents)**

In this task, an agent tries to collect yellow bananas and avoid the others. 

This section includes implementations of:

-   **Deep Q-Network (DQN)**
    
-   **Double DQN**
    
-   **Dueling DQN**
    
-   **Prioritized Experience Replay (PER)**
    

Used for classic control and discrete-action Unity environments.

----------

# 🏋️ **Project 2 – Continuous Control**

### **🎮 Environment: Reacher (Unity ML-Agents)**

In this task, a **20-agent Reacher** environment is used. Each agent attempts to maintain its arm at a target location.

### **🧠 Algorithm Used: PPO (Proximal Policy Optimization)**

I solved this environment using a **parallel multi-agent PPO implementation**, optimized for stability:

-   Clipped surrogate objective
    
-   Generalized Advantage Estimation (GAE)
    
-   Entropy bonus
    
-   Mini-batch SGD per epoch
    
-   Shared value & policy networks across all 20 agents
    

----------

### **🎯 Results**

✔️ **Environment Solved**  
Training converged to above +30 average score well before 200 episodes depending on seed.  
A “best” model checkpoint is saved automatically.

# 🤝 **Project 3 – Collaboration and Competition**

### **🎾 Environment: Tennis (2 agents)**

Two agents play tennis, trying to keep a ball in play.

### **🧠 Algorithm Used: MASAC (Multi-Agent Soft Actor-Critic)**

This project uses a **centralized critic + decentralized actors**:

-   Twin Q-networks (Double-Q SAC)
    
-   Soft Value function
    
-   Entropy regularization
    
-   Multi-agent replay buffer
    
-   Joint state-action critic
    
-   Independent policy updates per agent
    

The training includes checkpoint saving:

-   `best_checkpoint.pth` (highest avg score)
        
-   Optional `solved.pth` if score ≥ 0.5

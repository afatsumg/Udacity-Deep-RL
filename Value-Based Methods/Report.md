
# 🧠 **Report: Navigation (Banana Collector)**

## 📘 **Project Overview**

In this project, an agent is trained to collect yellow bananas in a large, square world while avoiding blue bananas.  
The goal of the agent is to maximize the total reward it receives.

-   **Reward +1** for collecting a yellow banana
    
-   **Reward -1** for collecting a blue banana
    
-   **Environment solved when:**  
    The agent achieves an **average score of +13 over 100 consecutive episodes**
    

----------

## ⚙️ **1. Environment Details**

Property

Description

**State space**

37 continuous values (agent’s velocity and ray-based perception of objects)

**Action space**

4 discrete actions: `0 = forward`, `1 = backward`, `2 = turn left`, `3 = turn right`

**Goal**

Collect yellow bananas, avoid blue ones

**Solved condition**

Average reward ≥ **+13** over 100 episodes

----------

## 🧩 **2. Learning Algorithm**

I implemented a **Dueling Double DQN with Prioritized Experience Replay (PER)** and **N-Step Returns**.

### 🔸 Base Algorithm: Deep Q-Network (DQN)

The Q-network estimates state-action values Q(s,a).  
At each step:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \big[ r_t + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \big]
$$


### 🔸 Double DQN

To reduce overestimation bias:

$$
Q_{\text{target}} = r + \gamma Q_{\text{target}}(s', \arg\max_a Q_{\text{local}}(s', a))
$$


### 🔸 Dueling Network Architecture

The Q-network is split into two streams:

$$
Q(s, a) = V(s) + \Bigg(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \Bigg)
$$


-   **Value stream (V):** Estimates how good it is to be in state `s`
    
-   **Advantage stream (A):** Estimates the relative benefit of each action
    

### 🔸 Prioritized Experience Replay (PER)

Transitions are sampled with probability proportional to their **TD-error**, giving more learning focus to surprising experiences.

### 🔸 N-Step Returns

Used n=3 to propagate reward information faster:

$$
G_t^{(n)} = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n Q(s_{t+n}, a_{t+n})
$$

----------

## ⚙️ **3. Model Architecture**

**Dueling Q-Network:**

`Input Layer:  37  (state  size)  Hidden Layer 1:  64 units,  ReLU  Hidden Layer 2:  64 units,  ReLU  Value Stream:  Linear(64 →  1)  Advantage Stream:  Linear(64  →  4)  Output:  Q-values  for  4  actions` 

----------

## 🔧 **4. Hyperparameters**


| Parameter | Value |
|--|--|
| Replay Buffer Size | 500,000 |
| Batch Size | 64 |
| Discount Factor (γ) | 0.99 |
| Learning Rate (LR) | 1e-4 |
| Soft Update (τ) | 5e-3 |
| Update Every | 4 steps |
| Epsilon Start | 1.0 |
| Epsilon End | 0.05 |
| Epsilon Decay | 0.999 |
| PER α | 0.6 |
| PER β | Annealed from 0.4 → 1.0 |
| N-Step | 3 |

----------

## 📈 **5. Results**

After integrating **Dueling + Double DQN + Prioritized Experience Replay + N-Step Returns**,  
the agent successfully solved the environment.

-   **Environment solved in:** ~700 episodes
    
-   **Best average:** 13.89
    
-   **Training Curve:**
    
![Training Performance](learning_graph.png)

----------

## 💡 **6. Ideas for Future Work**
    
**Rainbow DQN:** Combine all extensions (NoisyNet, PER, Dueling, N-Step, etc.).
 
----------

## 📜 **7. References**

-   Mnih et al., _Human-level control through deep reinforcement learning_ (Nature, 2015)
    
-   Wang et al., _Dueling Network Architectures for Deep Reinforcement Learning_ (2016)
    
-   Schaul et al., _Prioritized Experience Replay_ (2016)
    
-   Hessel et al., _Rainbow: Combining Improvements in Deep RL_ (2018)

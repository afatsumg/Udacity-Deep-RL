
# 🦾 Project 2: Continuous Control (Reacher Environment)

This repository contains the solution for the Udacity Deep Reinforcement Learning Nanodegree's second project: **Continuous Control**, using the **Reacher** environment. The solution employs the **Proximal Policy Optimization (PPO)** algorithm.

[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

## 1. Project Details

### Environment Overview

The **Reacher** environment features a double-jointed arm that must move to and track a dynamic target location. A reward of **+0.1** is provided for each time step the agent's hand is within the goal location.

This project specifically solves the **second version** of the environment, which contains **20 identical, parallel agents**.

### Trained Agent Performance

The agent was successfully trained to achieve the required score criterion.
---
### ⚙️ State and Action Spaces

| Space | Dimension | Type | Description |
| :--- | :--- | :--- | :--- |
| **Observation Space** | 33 (per agent) | Continuous | Corresponds to position, rotation, velocity, and angular velocities of the arm. |
| **Action Space** | 4 (per agent) | Continuous | Torques applicable to the two joints. Values must be in the **[-1, 1]** range. |

---
### ✅ Environment Solved Criterion (Multi-Agent Version)

The environment is considered solved when the agents achieve an **average score of +30** over 100 consecutive episodes.

The scoring is calculated as follows:

1.  After each episode, the reward received by all 20 agents is summed up.
2.  The average of these 20 scores is calculated, yielding the **episode average score**.
3.  The environment is solved when the 100-episode moving average of these episode average scores is **+30 or greater**.

---

## 2. Getting Started

To run the code and train the agent, you must install the dependencies and download the Unity environment file.

### 1. Dependencies

The project relies on standard scientific Python libraries and the Unity ML-Agents toolkit.

# Recommended environment setup
conda create --name drlnd python=3.9
conda activate drlnd

# Install core libraries
pip install numpy torch matplotlib unityagents

### 2. Downloading the Reacher Environment

Since this project solves the **Multi-Agent (20 Agents)** version, download the executable matching your operating system and place it in the repository's root folder.

## 3. Instructions

The PPO agent is implemented across `model.py` (Actor and Critic networks) and `ppo_agent.py` (PPO training logic).

### A. How to Train the Agent

Training is typically performed using the provided Jupyter Notebook (`Continuous_Control.ipynb`).

1.  Ensure the environment file path is correctly set within the notebook (e.g., `UnityEnvironment(file_name="Reacher_Windows_x86_64/Reacher.exe")`).
    
2.  Execute the cells sequentially to initialize the agent and start the training loop.
    
3.  The agent saves its best weights to **`ppo_reacher_solved.pth`** upon reaching the solving criterion ($+30.0$).
    

### B. How to Run the Trained Agent (Evaluation)

To observe the performance of the solved agent visually, you can use the evaluation code cell in the notebook or a dedicated script.

1.  Ensure the **`ppo_reacher_solved.pth`** file is present.
    
2.  Set `train_mode=False` in the environment reset call to enable the graphical display:
    
    Python
    
    ```
    env_info = env.reset(train_mode=False)[brain_name] 
    
    ```
    
3.  Load the weights and run the evaluation loop, ensuring that you load the checkpoint's weights into the `model.load_state_dict()` and use `model.eval()` for deterministic actions.

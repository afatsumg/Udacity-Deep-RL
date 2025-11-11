# utils.py
import numpy as np
import torch

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    """
    T = len(rewards)
    adv = np.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        nextvalue = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextvalue * nonterminal - values[t]
        adv[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    returns = adv + values
    return adv, returns

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)

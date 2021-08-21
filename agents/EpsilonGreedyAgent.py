"""
Epsilon greedy agent
"""

import random
import numpy as np
from .MABAgent import MABAgent

class EpsilonGreedyAgent(MABAgent):
    
    def __init__(self, n_arms, epsilon = 0.1, optimism=None):
        super(EpsilonGreedyAgent,self).__init__(n_arms, optimism)
        self.epsilon = epsilon
        self.optimism = optimism

    def step(self):
        """(Override) Returns the arm that should be pulled using epsilon-greedy"""
        averages = self.averages
        n_arms = self.n_arms
        if random.random() < self.epsilon:
            return random.randint(0, n_arms-1)
        else:
            return np.argmax(averages)

    def get_name(self):
        return f"Epsilon-Greedy MAB w/e={self.epsilon}, opt={self.optimism}"

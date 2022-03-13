"""
Epsilon greedy MAB agent.
"""

import random
import numpy as np
from .MABAgent import MABAgent

class EpsilonGreedyAgent(MABAgent):
    """
    Implements a multi armed bandit agent following the epsilon greedy policy.
    """   

    def __init__(self, n_arms, epsilon = 0.1, optimism=None):
        """
        Initializes epsilon_greedy Agent. 
        
        Args:
            n_arms: number of arms
            epsilon: probability of (random) exploration
            optimism: starting estimation for the value of every arm.
        """
        super(EpsilonGreedyAgent,self).__init__(n_arms, optimism)
        self.epsilon = epsilon
        self.optimism = optimism

    def step(self):
        """
        (Override) Returns the arm that should be pulled, using epsilon greedy.

        Returns:
            Index i of the arm that the policy decided to pull.
        """
        averages = self.averages
        n_arms = self.n_arms
        if random.random() < self.epsilon:
            arm = random.randint(0, n_arms-1)
            return arm
        else:
            arm = np.random.choice(np.flatnonzero(averages == averages.max()))# Random tie-breaking
            return arm

    def get_name(self):
        """
        String representation of the agent.

        Returns:
            string representing the agent.
        """
        return f"Epsilon-Greedy MAB w/e={self.epsilon}"

"""
EXP3 agent
"""

import numpy as np
from .MABAgent import MABAgent

def EXP3Gamma(best_value, n_rounds, n_arms):
    """
    Helper function to obtain "theoretically good" value for gamma
    best_value -> upper bound for the reward values
    n_rounds -> number of iterations for EXP3
    n_arms -> number of arms
    """
    g = best_value * n_rounds # Upper bound for final reward
    return np.min(np.array([1,np.sqrt(n_arms*np.log(n_arms)/(np.exp(1) * g))]))

class EXP3Agent(MABAgent):

    def __init__(self, n_arms, exploration_rate = 0.1, optimism=None):
        """
        exploration_rate -> weight (0 to 1) that is given to exploration vs exploitation.
        """
        super(EXP3Agent,self).__init__(n_arms, optimism)
        self.exprate = exploration_rate
        self.optimism = optimism
        self.weights = np.ones(n_arms) # EXP3 weights
        self.probs = np.empty(n_arms) # EXP3 probabilities

    def step(self):
        """(Override) Returns the arm that should be pulled using UCB"""

        # Compute each arms probabilities
        self.probs = (1-self.exprate)*(self.weights/sum(self.weights)) + self.exprate/self.n_arms

        return np.random.choice(range(self.n_arms), p=self.probs)

    def reward(self, n_arm, reward):
        """Updates the knowledge given the reward"""
        super().reward(n_arm, reward)

        #Update weights
        self.weights[n_arm] = self.weights[n_arm] * np.exp(self.exprate * reward / (self.n_arms * self.probs[n_arm]))

    def reset(self):
        super().reset()
        self.weights = np.ones(self.n_arms)
        self.probs = np.empty(self.n_arms)


    def get_name(self):
        return f"EXP3, gamma={self.exprate}, opt={self.optimism}"

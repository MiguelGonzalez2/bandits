"""
MAB Agent class.
"""

import numpy as np

class MABAgent():
    """Abstract class for MAB Agent"""
    def __init__(self, n_arms, optimism=None):
        """
        Initializes MABAgent object.
        n_arms -> Number of arms
        optimism -> starting value for rewards
        """
        self.averages = np.array([optimism if optimism else np.NINF] * n_arms)
        self.optimism = optimism
        self.times_explored = np.zeros(n_arms)
        self.steps = 0
        self.n_arms = n_arms

    def reward(self, n_arm, reward):
        """Updates the knowledge given the reward"""
        if self.averages[n_arm] == np.NINF:
            self.averages[n_arm] = reward
        else:
            old_value = self.averages[n_arm]
            # Update the average with the new observation
            self.averages[n_arm] = old_value + 1/(self.times_explored[n_arm]+1) * (reward - old_value)

        self.times_explored[n_arm] += 1

    def step(self):
        """Returns the arm that should be pulled. This should be overriden"""
        return 0

    def reset(self):
        """Resets agent."""
        self.averages = np.array([self.optimism if self.optimism else np.NINF] * self.n_arms)
        self.times_explored = np.zeros(self.n_arms)

    def get_best(self):
        """Returns the index of the best prediction so far"""
        return np.argmax(self.averages)

    def get_name(self):
        return "Default MAB"

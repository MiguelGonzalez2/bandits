"""
Generic Multi Armed Bandit Agent class.
Overriding this class allows for custom agent implementations.
"""

import numpy as np

class MABAgent():
    """Abstract class for MAB Agent"""

    def __init__(self, n_arms, optimism=None):
        """
        Initializes MABAgent object.
        Args:
            n_arms: Number of arms
            optimism: starting value for rewards
        """
        self.averages = np.array([optimism if optimism else np.NINF] * n_arms)
        self.optimism = optimism
        self.times_explored = np.zeros(n_arms)
        self.n_arms = n_arms
        self.is_dueling = False # Used when comparing DBs and MABs in the same simulation

    def reward(self, n_arm, reward):
        """
        Updates the knowledge given the reward. 

        Args:
            n_arm: pulled arm.
            reward: numerical reward obtained.
        """
        if self.averages[n_arm] == np.NINF:
            self.averages[n_arm] = reward
        else:
            old_value = self.averages[n_arm]
            # Update the average with the new observation
            self.averages[n_arm] = old_value + 1/(self.times_explored[n_arm]+1) * (reward - old_value)

        self.times_explored[n_arm] += 1

    def step(self):
        """
        (Override) Returns the arm that should be pulled, using EXP3.

        Returns:
            Index i of the arm that the policy decided to pull.
        """
        return 0

    def reset(self):
        """
        Fully resets the agent
        """
        self.averages = np.array([self.optimism if self.optimism else np.NINF] * self.n_arms)
        self.times_explored = np.zeros(self.n_arms)

    def get_best(self):
        """
        Get the best arm prediction so far.

        Returns:
            index of the estimated best arm.
        """
        return np.argmax(self.averages)

    def get_name(self):
        """
        String representation of the agent.

        Returns:
            string representing the agent.
        """
        return "Default MAB"

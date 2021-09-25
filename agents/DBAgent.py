"""
DB Agent class.
"""

import numpy as np

class DBAgent():
    """Abstract class for DB Agent"""
    def __init__(self, n_arms):
        """
        Initializes MABAgent object.
        n_arms -> Number of arms
        optimism -> starting value for rewards
        """
        self.outcomes = np.zeros((n_arms, n_arms)) # In position (i,j), # of times i beat j.
        self.n_arms = n_arms
        self.is_dueling = True # Used when comparing DBs and MABs in the same simulation

    def reward(self, n_arm_1, n_arm_2, one_wins):
        """
        Updates the knowledge given the reward. Since it's a Dueling Bandit, the reward
        is a boolean indicating whether the first arm wins or not.
        Override in children classes with the extra behaviour needed.
        """
        if not one_wins:
            n_arm_1, n_arm_2 = n_arm_2, n_arm_1
        self.outcomes[n_arm_1, n_arm_2] += 1

    def step(self):
        """Returns the arm that should be pulled. This should be overriden"""
        return 0

    def reset(self):
        """Resets agent."""
        self.outcomes = np.zeros((self.n_arms, self.n_arms))

    def get_ratio(self, n_arm_1, n_arm_2):
        """
        Returns the ratio of wins for n_arm_1 against n_arm_2
        """
        wins1 = self.outcomes[n_arm_1, n_arm_2]
        wins2 = self.outcomes[n_arm_2, n_arm_1]

        return wins1 / (wins1 + wins2)

    def get_comparison_count(self, n_arm_1, n_arm_2):
        """
        Returns the number of comparisons between the two arms specified
        """
        return self.outcomes[n_arm_1, n_arm_2] + self.outcomes[n_arm_2, n_arm_1]

    def get_total_comparison_count(self):
        """
        Returns the total comparison count.
        """
        return np.sum(self.outcomes)
        
    def get_name(self):
        return "Default DB"

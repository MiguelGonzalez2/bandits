"""
Generic Dueling Bandit Agent class.
Overriding this class allows for custom agent implementations.
"""

import numpy as np

class DBAgent():
    """Abstract class for DB Agent"""

    def __init__(self, n_arms):
        """
        Initializes MABAgent object.

        Args:
            n_arms: Number of arms
            optimism: starting value for rewards
        """

        self.outcomes = np.zeros((n_arms, n_arms)) # In position (i,j), # of times i beat j.
        self.n_arms = n_arms
        self.is_dueling = True # Used when comparing DBs and MABs in the same simulation

    def reward(self, n_arm_1, n_arm_2, one_wins):
        """
        Updates the knowledge given the reward. Since it's a Dueling Bandit, the reward
        is a boolean indicating whether the first arm wins or not.

        Args:
            n_arm_1: first arm of the pulled pair.
            n_arm_2: second arm of the pulled pair.
            one_wins: boolean indicating whether the first arm won.
        """

        if not one_wins:
            n_arm_1, n_arm_2 = n_arm_2, n_arm_1
        self.outcomes[n_arm_1, n_arm_2] += 1

    def step(self):
        """
        Returns the pair that should be matched. Override to use custom policy.

        Returns:
            Pair of indices (i,j) that the policy decided to pull.
        """

        return (0,0)

    def reset(self):
        """
        Fully resets the agent
        """

        self.outcomes = np.zeros((self.n_arms, self.n_arms))

    def get_ratio(self, n_arm_1, n_arm_2):
        """
        Returns the ratio of wins for n_arm_1 against n_arm_2.

        Args:
            n_arm_1: First arm to compare.
            n_arm_2: Second arm to compare.
        
        Returns:
            ratio of wins for n_arm_1 against n_arm_2, that is, empirical
            estimation of the probability that n_arm_1 beats n_arm_2. Defaults
            to 1/2 if no matches have been recorded.
        """

        wins1 = self.outcomes[n_arm_1, n_arm_2]
        wins2 = self.outcomes[n_arm_2, n_arm_1]

        if wins1 + wins2 > 0:
            return wins1 / (wins1 + wins2)
        else:
            return 1/2

    def get_comparison_count(self, n_arm_1, n_arm_2):
        """
        Returns the number of comparisons between the two arms specified.

        Args:
            n_arm_1: First arm to compare.
            n_arm_2: Second arm to compare.
        
        Returns:
            number of matches recorded between n_arm_1 and n_arm_2 (independent of order).
        """
        return self.outcomes[n_arm_1, n_arm_2] + self.outcomes[n_arm_2, n_arm_1]

    def get_total_comparison_count(self):
        """
        Gets the total comparison count.

        Returns:
            total matches recorded.
        """
        return np.sum(self.outcomes)

    def get_name(self):
        """
        String representation of the agent.

        Returns:
            string representing the agent.
        """
        return "Default DB"

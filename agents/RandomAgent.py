"""
Random DB Agent to serve as baseline.
"""

import random
from .DBAgent import DBAgent

class RandomAgent(DBAgent):
    """
    Implements a dueling bandit agent following the random policy.
    """

    def step(self):
        """
        (Override) Returns the pair that should be matched, randomly.

        Returns:
            Pair of indices (i,j) that the policy decided to pull.
        """
        arm1 = random.randint(0, self.n_arms-1)
        arm2 = random.randint(0, self.n_arms-1)
        return arm1, arm2

    def get_name(self):
        """
        String representation of the agent.

        Returns:
            string representing the agent.
        """
        return f"Random DB"

"""
UCB multi armed bandit agent.
"""

import numpy as np
from .MABAgent import MABAgent

class UCBAgent(MABAgent):
    """
    Implements a multi armed bandit agent following the UCB policy.
    """  
    def __init__(self, n_arms, exploration_rate = 1, optimism=None):
        """
        Initializes the agent.

        Args:
            n_arms: number of arms.
            exploration_rate: weight that is given to uncertainty vs average 
                The higher the rate, the more exploration occurs.
            optimism: starting estimation for the value of each arm.
        """
        super(UCBAgent,self).__init__(n_arms, optimism)
        self.exprate = exploration_rate
        self.optimism = optimism

    def step(self):
        """
        (Override) Returns the arm that should be pulled, using UCB.

        Returns:
            Index i of the arm that the policy decided to pull.
        """

        # Compute "time", that is the order of this step
        time = sum(self.times_explored) + 1

        ucb_scores = np.zeros(self.n_arms)

        for i in range(self.n_arms):
            # If arm hasnt been explored, explore.
            if self.times_explored[i] == 0:
                return i
            else:
                # Compute ucb score
                ucb_scores[i] = self.averages[i]  + self.exprate * np.sqrt(np.log(time)/self.times_explored[i])

        # Pull the best score
        return np.argmax(ucb_scores)


    def get_name(self):
        """
        String representation of the agent.

        Returns:
            string representing the agent.
        """
        return f"UCB, c={self.exprate}"

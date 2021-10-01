"""
UCB agent
"""

import numpy as np
from .MABAgent import MABAgent

class UCBAgent(MABAgent):

    def __init__(self, n_arms, exploration_rate = 1, optimism=None):
        """
        exploration_rate -> weight that is given to uncertainty vs average 
        The higher the rate, the more exploration occurs.
        """
        super(UCBAgent,self).__init__(n_arms, optimism)
        self.exprate = exploration_rate
        self.optimism = optimism

    def step(self):
        """(Override) Returns the arm that should be pulled using UCB"""

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
        return f"UCB, c={self.exprate}"

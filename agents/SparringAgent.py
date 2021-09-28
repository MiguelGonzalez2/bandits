"""
Sparring (MAB reduction) dueling bandit agent.
"""

import random
import numpy as np
from numpy.core.numeric import Inf
from .DBAgent import DBAgent

class SparringAgent(DBAgent):
    
    def __init__(self, n_arms, mab1, mab2):
        """
        Initializes Sparring agent. This agent allows a MAB to be used
        with dueling bandits.
        mab1 -> MAB in charge of arm 1. Must be of type MABAgent.
        mab2 -> MAB in charge of arm 2. Must be of type MABAgent.
        """
        super(SparringAgent,self).__init__(n_arms)
        
        self.mab1 = mab1
        self.mab2 = mab2

        self.mab1.reset()
        self.mab2.reset()

    def reward(self, n_arm_1, n_arm_2, one_wins):
        """
        Updates the knowledge given the reward. Since it's a Dueling Bandit, the reward
        is a boolean indicating whether the first arm wins or not.
        In Sparring, it feeds each result to the respective MAB agent.
        """
        self.mab1.reward(n_arm_1, int(one_wins))
        self.mab2.reward(n_arm_2, int(not one_wins))

    def step(self):
        """(Override) Returns the pair that should be matched, using Sparring"""

        return self.mab1.step(), self.mab2.step()

        
    def reset(self):
        """Fully resets the agent"""
        super().reset()
        self.mab1.reset()
        self.mab2.reset()

    def get_name(self):
        name1 = self.mab1.get_name()
        name2 = self.mab2.get_name()
        return f"Sparring DB: {name1} VS {name2 if name1 != name2 else 'same'}"

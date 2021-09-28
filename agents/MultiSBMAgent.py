"""
MultiSBM (MAB reduction) dueling bandit agent.
"""

import random
import numpy as np
from numpy.core.numeric import Inf
from .DBAgent import DBAgent

class MultiSBMAgent(DBAgent):
    
    def __init__(self, n_arms, mab_callable, mab_args=[], mab_kwargs=dict()):
        """
        Initializes MultiSBM agent. This agent allows a MAB to be used
        with dueling bandits.
        mab_callable -> Callable that returns an object of type MABAgent
        to be used within the MultiSBM.
        mab_args -> List of arguments to be passed to the MAB callable on
        creation.
        mab_kwargs -> Dict of key-word arguments to be passed to the MAB
        callable on creation.
        """
        super(MultiSBMAgent,self).__init__(n_arms)
        self.mab_callable = mab_callable

        # Create one MAB per arm. Each MAB will face the arm it's indexed with
        self.mabs = [mab_callable(*mab_args, **mab_kwargs) for _ in range(n_arms)]

        # Ensure the MABs are fresh.
        for mab in self.mabs:
            mab.reset()

        # Last played arm by a MAB
        self.last_played = 0

    def reward(self, n_arm_1, n_arm_2, one_wins):
        """
        Updates the knowledge given the reward. Since it's a Dueling Bandit, the reward
        is a boolean indicating whether the first arm wins or not.
        In MultiSBM it feeds the comparison result to the agent in charge of facing arm 1.
        """
        
        # Feed the MAB whether it won.
        self.mabs[n_arm_1].reward(n_arm_2, int(not one_wins))

    def step(self):
        """(Override) Returns the pair that should be matched, using MultiSBM"""

        # Arm 1 is determined by the last arm 2
        arm1 = self.last_played

        # Arm 2 is determined by the MAB in charge of playing against arm 1
        arm2 = self.mabs[arm1].step()
        self.last_played = arm2

        return arm1, arm2

        
    def reset(self):
        """Fully resets the agent"""
        super().reset()
        for mab in self.mabs:
            mab.reset()
        self.last_played = 0

    def get_name(self):
        return f"MultiSBM DB w/MAB: {self.mabs[0].get_name()}"

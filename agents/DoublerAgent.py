"""
Doubler (MAB reduction) dueling bandit agent.
"""

import random
import numpy as np
from numpy.core.numeric import Inf
from .DBAgent import DBAgent

class DoublerAgent(DBAgent):
    
    def __init__(self, n_arms, mab):
        """
        Initializes Doubler agent. This agent allows a MAB to be used
        with dueling bandits.
        mab -> Object of type MABAgent that will be used for doubler.
        """
        super(DoublerAgent,self).__init__(n_arms)
        self.mab = mab

        # Ensure the MAB is fresh.
        self.mab.reset()

        # Counter of epochs, since doubler has to do more operations every
        # once in a while (whenever a "epoch" finishes).
        self.epoch = 1

        # Counter of steps within the epoch
        self.steps = 1

        # Keeps track of the arms played by the MAB.
        self.played = set()

        # Valid moves for the MAB opponent. Gets updated over time.
        # Used a list instead of a set for random choice performance.
        self.opponent = [0]

    def reward(self, n_arm_1, n_arm_2, one_wins):
        """
        Updates the knowledge given the reward. Since it's a Dueling Bandit, the reward
        is a boolean indicating whether the first arm wins or not.
        In Doubler, it feeds the result to the MAB. We assume in this implementation that
        the arm played by the MAB is arm 1.
        """
        
        # Feed the MAB whether it won
        self.mab.reward(n_arm_1, int(one_wins))

        self.steps += 1

        # If the epoch is finished, start a new one.
        if self.steps > 2**self.epoch:
            self.steps = 1
            self.epoch += 1
            self.opponent = list(self.played)
            self.played = set()

    def step(self):
        """(Override) Returns the pair that should be matched, using DB"""

        # Arm 1 is played by the MAB
        arm1 = self.mab.step()
        self.played.add(arm1)

        # Arm 2 is played by the uniform sampler depending on past epoch
        arm2 = random.choice(self.opponent)

        return arm1, arm2

        
    def reset(self):
        """Fully resets the agent"""
        super().reset()
        self.mab.reset()
        self.epoch = 1
        self.steps = 1
        self.played = set()
        self.opponent = [0]

    def get_name(self):
        return f"Doubler DB w/MAB: {self.mab.get_name()}"

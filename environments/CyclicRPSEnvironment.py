"""
Environment with a cyclic, rock-paper-scissors-like distribution.
"""

import numpy as np
from .Environment import Environment
from random import random

class CyclicRPSEnvironment(Environment):
    """
    Implements cyclic environment without condorcet winner.
    IMPORTANT: this environment behaves exactly like a gaussian environment
    if only the method "step" is used to get rewards. However, the method
    "dueling_step" is the one that will return the appropriate outcome with
    cyclic distribution.
    """

    def __init__(self, n_arms, value_generator = np.random.normal, winner_prob=2/3):
        """
        Initializes the environment.
        n_arms -> Number of arms
        winner_prob -> probability that i beats j in the case that i is better than j.
        value_generator -> Generator function for each arm hidden value (true reward)
        """
        super(CyclicRPSEnvironment,self).__init__(n_arms, value_generator)
        self.winner_prob = winner_prob

    def dueling_step(self, n_arm1, n_arm2):
        """
        Returns rewards for a pair of arms, updating internal values.
        This is used for Dueling Bandits steps. This environment returns
        values 1 and 0 as rewards, to compare dueling algorithms only, with
        cyclic probability distribution.
        """
        self.pulls[n_arm1] += 1
        self.pulls[n_arm2] += 1
        self.steps += 1

        # Check if the first one should have the winning probability
        first_strong = ((n_arm1 - n_arm2) % self.n_arms) < self.n_arms/2

        # fix the cases where arm1 - arm2 = self.n_arms/2 in case n_arms is even
        if ((n_arm1 - n_arm2) % self.n_arms) == self.n_arms/2:
            first_strong = n_arm1 < n_arm2

        # Assign 1 to the winner, 0 to the loser
        first = 0
        second = 1
        random_value = random()

        if first_strong and random_value < self.winner_prob:
            # First wins with strong probability
            first, second = second, first
        elif not first_strong and random_value < (1 - self.winner_prob):
            # First wins with weak probability
            first, second = second, first

        return (first, second)

    def get_probability_dueling(self, arm1, arm2):
        """
        Receives two arms and returns the probability that arm1 >= arm2.
        The "generalized rock paper scissors" method is applied.
        """
        probability = self.winner_prob if ((arm1 - arm2) % self.n_arms) < self.n_arms/2 else (1 - self.winner_prob)
        # fix the cases where arm1 - arm2 = self.n_arms/2 in case n_arms is even
        if ((arm1 - arm2) % self.n_arms) == self.n_arms/2:
            probability = self.winner_prob if arm1 < arm2 else (1-self.winner_prob)
        return probability if arm1 != arm2 else 1/2

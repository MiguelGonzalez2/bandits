"""
Environment with a cyclic, rock-paper-scissors-like distribution.

Only meaningful upon pairwise comparisons (behaves the same as GaussianEnvironment for single pulls)
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

    def __init__(self, n_arms, value_generator = np.random.normal, values = None, winner_prob=2/3, std=0):
        """
        Initializes the environment.

        Args:
            n_arms: Number of arms
            winner_prob: probability that i beats j in the case that i is better than j.
            value_generator: Generator function for each arm hidden value (true reward)
            values: Actual arm values. If given, value_generator is unused.
            variance: Random noise from gaussian(0,std) is applied to every probability, then clipped
                to either [1/2,1] or [0,1/2] depending on whether it's the winner or the loser.
        """
        super(CyclicRPSEnvironment,self).__init__(n_arms, value_generator, values)
        self.probabilities = np.zeros((n_arms, n_arms)) # Entry [i,j] is probability that i beats j
        self.winner_prob = winner_prob
        self.std = std

        # Initialize the distribution table
        for arm1 in range(n_arms):
            for arm2 in range(arm1+1):
                if arm1 == arm2:
                    self.probabilities[arm1,arm2] = 1/2
                    self.probabilities[arm2,arm1] = 1/2
                else:
                    # Compute the probability of winning
                    prob = np.clip(winner_prob + np.random.normal(loc=0, scale=std),1/2,1)
                    # These are the two win conditions, if any is met, arm1 wins.
                    win1 = ((arm1 - arm2) % self.n_arms) < self.n_arms/2
                    win2 = (((arm1 - arm2) % self.n_arms) == self.n_arms/2 and arm1 < arm2)
                    if win1 or win2:
                        self.probabilities[arm1, arm2] = prob
                        self.probabilities[arm2, arm1] = 1-prob
                    else:
                        self.probabilities[arm1, arm2] = 1-prob
                        self.probabilities[arm2, arm1] = prob

        # Update the copeland scores
        self.copeland_scores = np.count_nonzero(self.probabilities > 1/2, axis=1)/(self.n_arms-1)


    def dueling_step(self, n_arm1, n_arm2):
        """
        Returns rewards for a pair of arms, updating internal values.
        This is used for Dueling Bandits steps. This environment returns
        values 1 and 0 as rewards, to compare dueling algorithms only, with
        cyclic probability distribution.

        Args:
            n_arm1: first arm of the pair.
            n_arm2: second arm of the pair.

        Returns:
            rewards for each bandit of the pair, being 1 for the winner and 0
            for the loser.
        """
        self.pulls[n_arm1] += 1
        self.pulls[n_arm2] += 1
        self.steps += 1

        if random() < self.probabilities[n_arm1, n_arm2]:
            # First wins
            return (1, 0)

        return (0, 1)

    def reset(self):
        """
        Resets environment internals
        """
        super().reset()

        # Initialize the distribution table
        for arm1 in range(self.n_arms):
            for arm2 in range(arm1+1):
                if arm1 == arm2:
                    self.probabilities[arm1,arm2] = 1/2
                    self.probabilities[arm2,arm1] = 1/2
                else:
                    # Compute the probability of winning
                    prob = np.clip(self.winner_prob + np.random.normal(loc=0, scale=self.std),1/2,1)
                    # These are the two win conditions, if any is met, arm1 wins.
                    win1 = ((arm1 - arm2) % self.n_arms) < self.n_arms/2
                    win2 = (((arm1 - arm2) % self.n_arms) == self.n_arms/2 and arm1 < arm2)
                    if win1 or win2:
                        self.probabilities[arm1, arm2] = prob
                        self.probabilities[arm2, arm1] = 1-prob
                    else:
                        self.probabilities[arm1, arm2] = 1-prob
                        self.probabilities[arm2, arm1] = prob

        # Update the copeland scores
        self.copeland_scores = np.count_nonzero(self.probabilities > 1/2, axis=1)/(self.n_arms-1)
        

    def get_probability_dueling(self, arm1, arm2):
        """
        Receives two arms and returns the probability that arm1 >= arm2.
        The "generalized rock paper scissors" method is applied.

        Args:
            arm1: first arm to be compared
            arm2: second arm to be compared

        Returns:
            "Probability" that arm1 >= arm2 (slightly modified so that P(arm1>=arm2) + P(arm2>=arm1) = 1)
        """
        return self.probabilities[arm1, arm2]


    def get_name(self):
        """
        String representation of the environment.

        Returns:
            string representing the environment.
        """
        return f"Noisy rock-paper-scissors arms with wp={self.winner_prob}, std={self.std}"
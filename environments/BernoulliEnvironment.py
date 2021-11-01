"""
Gaussian Arms Environment.
"""

from .Environment import Environment
import numpy as np

class BernoulliEnvironment(Environment):

    def __init__(self, n_arms):
        """
        Override the abstract constructor so that probabilities are in the correct range.
        """
        super(BernoulliEnvironment, self).__init__(n_arms, np.random.random)

    def pull(self, n_arm):
        """
        Pulls a given arm with a gaussian distribution centered @ value.
        """
        value = self.arms[n_arm]
        return int(np.random.random() < value)

    def get_probability_dueling(self, arm1, arm2):
        """
        Receives two arms and returns the probability that arm1 >= arm2.
        This should be overriden depending on the "pull" function, to match
        the distribution. 

        For the bernoulli distribution, its p1 * (1-p2) because we need arm1 = 1
        and arm2 = 0.
        """
        return self.arms[arm1] * (1-self.arms[arm2])

    def get_name(self):
        return f"Bernoulli arms"
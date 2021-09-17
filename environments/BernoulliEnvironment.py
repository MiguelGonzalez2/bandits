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
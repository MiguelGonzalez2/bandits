"""
Gaussian Arms Environment.
"""

from .Environment import Environment
import numpy as np

class GaussianEnvironment(Environment):

    def pull(self, n_arm):
        """
        Pulls a given arm with a gaussian distribution centered @ value.
        """
        value = self.arms[n_arm]
        return value + np.random.normal()
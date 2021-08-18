"""
Gaussian Arms Environment.
"""

from .MABEnvironment import MABEnvironment
import numpy as np

class GaussianEnvironment(MABEnvironment):

    def pull(self, n_arm):
        """
        Pulls a given arm with a gaussian distribution centered @ value.
        """
        value = self.arms[n_arm]
        return value + np.random.normal()
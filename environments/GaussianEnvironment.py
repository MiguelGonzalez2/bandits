"""
Gaussian Arms Environment.
"""

from .Environment import Environment
import numpy as np
from scipy.stats import norm

class GaussianEnvironment(Environment):

    def pull(self, n_arm):
        """
        Pulls a given arm with a gaussian distribution centered @ value.
        """
        value = self.arms[n_arm]
        return value + np.random.normal()

    def get_probability_dueling(self, arm1, arm2):
        """
        Receives two arms and returns the probability that arm1 >= arm2.
        This should be overriden depending on the "pull" function, to match
        the distribution. 

        For the normal distribution, is 1 - cdf((m2-m1)/sqrt(var1+var2))
        """
        return 1 - norm.cdf((self.arms[arm2] - self.arms[arm1]) / np.sqrt(2))
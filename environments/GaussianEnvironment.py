"""
Gaussian Arms Environment.
"""

from .Environment import Environment
import numpy as np
from scipy.stats import norm

class GaussianEnvironment(Environment):
    """
    Implements gaussian-distributed arm environment.
    """

    def pull(self, n_arm):
        """
        Pulls a given arm with a gaussian distribution with mean given by arm value.

        Args:
            n_arm: arm index to be pulled.
        
        Returns:
            numerical reward obtained.
        """
        value = self.arms[n_arm]
        return value + np.random.normal()

    def get_probability_dueling(self, arm1, arm2):
        """
        Receives two arms and returns the probability that arm1 >= arm2.
        This should be overriden depending on the "pull" function, to match
        the distribution. 

        For the normal distribution, is 1 - cdf((m2-m1)/sqrt(var1+var2))

        Args:
            arm1: first arm to be compared
            arm2: second arm to be compared

        Returns:
            Probability that arm1 >= arm2.
        """
        return 1 - norm.cdf((self.arms[arm2] - self.arms[arm1]) / np.sqrt(2))

    def get_name(self):
        """
        String representation of the environment.

        Returns:
            string representing the environment.
        """
        return f"Gaussian Arms"
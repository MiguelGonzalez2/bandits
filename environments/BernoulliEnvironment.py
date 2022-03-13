"""
Bernoulli-distributed Arms Environment.
"""

from .Environment import Environment
import numpy as np

class BernoulliEnvironment(Environment):
    """
    Implements bernoulli-distributed arm environment.
    """

    def __init__(self, n_arms):
        """
        Initializes the environment.

        Args:
            n_arms: number of arms.
        """
        super(BernoulliEnvironment, self).__init__(n_arms, np.random.random)

    def pull(self, n_arm):
        """
        Pulls a given arm with a bernoulli distribution with mean given by arm value.

        Args:
            n_arm: arm index to be pulled.
        
        Returns:
            numerical reward obtained.
        """
        value = self.arms[n_arm]
        return int(np.random.random() < value)

    def get_probability_dueling(self, arm1, arm2):
        """
        Receives two arms and returns the probability that arm1 >= arm2.
        This should be overriden depending on the "pull" function, to match
        the distribution. 

        For the bernoulli distribution, we use the probability that arm1 > arm2,
        plus the probability that arm1==arm2 HALVED, since we assume ties are broken
        randomly. This also guarantees that if E[arm1] = E[arm2], 1/2 is returned.

        Args:
            arm1: first arm to be compared
            arm2: second arm to be compared

        Returns:
            "Probability" that arm1 >= arm2 (slightly modified so that P(arm1>=arm2) + P(arm2>=arm1) = 1)
        """
        p1 = self.arms[arm1]
        p2 = self.arms[arm2]
        return p1*(1-p2) + (p1*p2 + (1-p1)*(1-p2))/2

    def get_name(self):
        """
        String representation of the environment.

        Returns:
            string representing the environment.
        """
        return f"Bernoulli arms"
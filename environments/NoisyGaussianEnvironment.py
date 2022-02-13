"""
Noisy Gaussian Arms Environment.
In this environment, when pulling dueling bandits, each pair has a 
random noise added so as to break transitivity.
"""

from .GaussianEnvironment import GaussianEnvironment
import numpy as np
from scipy.stats import norm

class NoisyGaussianEnvironment(GaussianEnvironment):

    def __init__(self, n_arms, value_generator = np.random.normal, values = None, d=0.1):
        """
        Initializes the environment.
        n_arms -> Number of arms
        value_generator -> Generator function for each arm hidden value (true reward)
        values -> Actual arm values. If given, value_generator is unused.
        d -> Amount of noise added. The higher, the less transitivity.
        """
        super(NoisyGaussianEnvironment,self).__init__(n_arms, value_generator, values)
        self.d = d
        self.epsilons = np.tril(np.random.normal(loc=0, scale=d**2, size=(n_arms, n_arms)), k=-1)
        t = np.transpose(self.epsilons)
        self.epsilons = self.epsilons + -t

        # Update the copeland scores
        self.copeland_scores = np.count_nonzero(self.epsilons - self.arms + self.arms[:, np.newaxis] > 0, axis=1)/(self.n_arms-1)


    def dueling_step(self, n_arm1, n_arm2):
        """
        Returns rewards for a pair of arms, updating internal values.
        This is used for Dueling Bandits steps. Note that the actual
        rewards are returned so that numerical metrics can be computed
        in order to compare with multi-armed bandits. HOWEVER, dueling
        bandits should never see these values, only the result of the
        pairwise comparison.
        """
        self.pulls[n_arm1] += 1
        self.pulls[n_arm2] += 1
        self.steps += 1
        value1 = self.arms[n_arm1]
        value2 = self.arms[n_arm2]
        epsilon = self.epsilons[n_arm1, n_arm2]
        # We add the epsilon value to the first arm to produce noise.
        return (value1 + np.random.normal() + epsilon, value2 + np.random.normal())


    def get_probability_dueling(self, arm1, arm2):
        """
        Receives two arms and returns the probability that arm1 >= arm2.
        This should be overriden depending on the "pull" function, to match
        the distribution. 

        For the normal distribution, is 1 - cdf((m2-m1)/sqrt(var1+var2))
        """
        return 1 - norm.cdf((self.arms[arm2] - self.arms[arm1] - self.epsilons[arm1,arm2]) / np.sqrt(2))


    def reset(self):
        """
        Resets environment internals
        """
        super().reset()
        self.epsilons = np.tril(np.random.normal(loc=0, scale=self.d**2, size=(self.n_arms, self.n_arms)), k=-1)
        t = np.transpose(self.epsilons)
        self.epsilons = self.epsilons + -t
        # Update the copeland scores
        self.copeland_scores = np.count_nonzero(self.epsilons - self.arms + self.arms[:, np.newaxis] > 0, axis=1)/(self.n_arms-1)


    def get_name(self):
        return f"Noisy Gaussian Arms with d={self.d}"
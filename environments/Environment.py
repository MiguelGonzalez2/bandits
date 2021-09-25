"""
Environment class.
"""

import numpy as np

class Environment():
    """Implements abstract MABEnvironment w/ constant output"""

    def __init__(self, n_arms, value_generator = np.random.normal):
        """
        Initializes the environment.
        n_arms -> Number of arms
        value_generator -> Generator function for each arm hidden value (true reward)
        """
        self.value_generator = value_generator
        self.arms = np.array([value_generator() for i in range(n_arms)])
        self.n_arms = n_arms
        self.pulls = np.zeros(n_arms) # Individual pull values
        self.steps = 0 # Total Steps

    def pull(self, n_arm):
        """
        Pulls a given arm and returns reward. Override in subclasses.
        """
        return self.arms[n_arm]

    def step(self, n_arm):
        """
        Returns reward for given arm, updating internal values
        This is used for Multi Armed Bandits steps.
        """
        self.pulls[n_arm] += 1
        self.steps += 1
        return self.pull(n_arm)

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
        return (self.pull(n_arm1), self.pull(n_arm2))

    def soft_reset(self):
        """
        Only resets metrics but environment is kept the same
        """
        self.pulls = np.zeros(self.n_arms)
        self.steps = 0
        
    def reset(self):
        """
        Resets environment internals
        """
        self.soft_reset()
        self.arms = np.array([self.value_generator() for i in range(self.n_arms)])

    def get_optimal(self):
        return np.argmax(self.arms)

    def get_optimal_value(self):
        return np.max(self.arms)

    def get_probability_dueling(self, arm1, arm2):
        """
        Receives two arms and returns the probability that arm1 >= arm2.
        This should be overriden depending on the "pull" function, to match
        the distribution. 
        """
        return arm1 / (arm1 + arm2)
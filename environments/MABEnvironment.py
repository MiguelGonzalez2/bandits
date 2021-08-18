"""
Environment class.
"""

import numpy as np

class MABEnvironment():
    """Implements abstract MABEnvironment w/ constant output"""

    def __init__(self, n_arms, value_generator = (lambda : np.random.normal())):
        """
        Initializes the environment.
        n_arms -> Number of arms
        value_generator -> Generator function for each arm hidden value (true reward)
        """
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
        """
        self.pulls[n_arm] += 1
        self.steps += 1
        return self.pull(n_arm)

    def get_optimal(self):
        return np.argmax(self.arms)

    def get_optimal_value(self):
        return np.max(self.arms)
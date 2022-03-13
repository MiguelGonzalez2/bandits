"""
Thompson Sampling Agent intended for bernoulli environments.
As such, it uses the beta distribution as its parameter distribution.
As presented in https://papers.nips.cc/paper/4321-an-empirical-evaluation-of-thompson-sampling.
"""

import numpy as np
from .MABAgent import MABAgent

class ThompsonBetaAgent(MABAgent):
    """
    Implements a multi armed bandit agent following the Thompson Sampling (with beta prior) policy.
    """    

    def __init__(self, n_arms, alpha_zero=1, beta_zero=1, failure_thres=1/2, optimism=None):
        """
        Initializes the agent. 

        Args:
            n_arms: number of arms.
            alpha_zero: starting alpha parameter value for the alpha distribution.
            beta_zero: starting beta parameter value for the beta distribution.
            failure_thres: governs what counts as a bernoulli failure. Any reward
                below the threshold will counted as a failure. This exists mainly to support
                testing this agent in non-bernoulli environments (don't set this value for bernoulli environments).
            optimism: starting estimation for the value of every arm.
        """
        super(ThompsonBetaAgent,self).__init__(n_arms, optimism)
        self.alpha = alpha_zero
        self.beta = beta_zero
        self.failure_thres = failure_thres
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)

    def step(self):
        """
        (Override) Returns the arm that should be pulled, using Thompson Sampling with beta prior.

        Returns:
            Index i of the arm that the policy decided to pull.
        """
        estimated_params = np.empty(self.n_arms) # Holds the estimated parameters

        # Estimate the parameters
        for arm in range(self.n_arms):
            estimated_params[arm] = np.random.beta(self.successes[arm] + self.alpha, self.failures[arm] + self.beta)

        # Return the arm which was estimated to be best.
        return np.argmax(estimated_params)

    def reward(self, n_arm, reward):
        """
        Updates the knowledge given the reward. 

        Args:
            n_arm: pulled arm.
            reward: numerical reward obtained.
        """
        super().reward(n_arm, reward)

        # Update counters
        if reward < self.failure_thres:
            self.failures[n_arm] += 1
        else:
            self.successes[n_arm] += 1

    def reset(self):
        """
        Fully resets the agent
        """
        super().reset()
        self.successes = np.zeros(self.n_arms)
        self.failures = np.zeros(self.n_arms)
        
    def get_name(self):
        """
        String representation of the agent.

        Returns:
            string representing the agent.
        """
        return f"Thompson Sampling MAB using Beta prior w/alpha={self.alpha}, beta={self.beta}"
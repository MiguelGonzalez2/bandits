"""
Double Thompson Sampling (DTS) Dueling Bandit Agent
"""

import random
import numpy as np
from numpy.core.defchararray import lower, upper
from numpy.core.fromnumeric import transpose
from numpy.core.numeric import Inf
from .DBAgent import DBAgent

class DTSAgent(DBAgent):
    
    def __init__(self, n_arms, alpha=1, beta=1, gamma=1):
        """
        Initializes Double Thompson Sampling agent. 
        alpha -> Starting alpha parameter for thompson sampling
        beta -> Starting beta parameter for thompson sampling
        gamma -> Size of the confidence interval for the starting UCB-like pruning phase.
        """
        super(DTSAgent,self).__init__(n_arms)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Time step
        self.time = 1

    def step(self):
        """(Override) Returns the pair that should be matched, using DTS"""

        # Confidence interval for each probability
        total_matches = self.outcomes + np.transpose(self.outcomes)
        mask = (total_matches != 0) # This will prevent division by zero, setting 1 in those places instead.
        conf_interval_sizes = np.sqrt(self.gamma * np.where(mask, np.divide(np.log(self.time), total_matches, where=mask), 1))
        upper_bounds = np.where(mask, np.divide(self.outcomes, total_matches, where=mask), 1) + conf_interval_sizes
        lower_bounds = np.where(mask, np.divide(self.outcomes, total_matches, where=mask), 1) - conf_interval_sizes
        np.fill_diagonal(upper_bounds, 1/2)
        np.fill_diagonal(lower_bounds, 1/2)
        
        # Copeland scores to discard losers
        exclude_diagonal = np.full((self.n_arms, self.n_arms), True)
        np.fill_diagonal(exclude_diagonal, False)
        scores = np.count_nonzero(np.logical_and(upper_bounds > 1/2, exclude_diagonal), axis=1)
        winners = (scores == scores.max())

        # Thompson sampling
        thetas = np.empty((self.n_arms, self.n_arms)) # Estimates for each probability

        for j in range(self.n_arms):
            for i in range(j):
                sample = np.random.beta(self.outcomes[i,j] + self.alpha, self.outcomes[j,i] + self.beta)
                thetas[i,j] = sample
                thetas[j,i] = 1 - sample

        # Select overall winner by updating scores using the sampled probabilities
        scores = np.where(winners, np.count_nonzero(np.logical_and(thetas > 1/2, exclude_diagonal), axis=1), np.NINF)
        arm1 = np.random.choice(np.flatnonzero(scores == scores.max()))

        # Update theta scores
        for i in range(self.n_arms):
            if i == arm1:
                thetas[i,i] = 1/2
            else:
                thetas[i,arm1] = np.random.beta(self.outcomes[i,arm1] + self.alpha, self.outcomes[arm1,i] + self.beta)

        # Select competitor as follows: pick the best one from the "uncertain" pairs.
        uncertain_pairs = np.where(lower_bounds[:, arm1] <= 1/2, thetas[:, arm1], np.NINF)
        arm2 = np.random.choice(np.flatnonzero(uncertain_pairs == uncertain_pairs.max()))

        self.time += 1
        return arm1, arm2
        
    def reset(self):
        """Fully resets the agent"""
        super().reset()
        self.time = 1

    def get_name(self):
        return f"DTS DB w/a: {self.alpha}, b: {self.beta}, g: {self.gamma}"

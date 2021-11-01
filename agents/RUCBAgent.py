"""
Relative UCB (RUCB) Dueling Bandit Agent
"""

import numpy as np
from .DBAgent import DBAgent

class RUCBAgent(DBAgent):
    
    def __init__(self, n_arms, alpha=1):
        """
        Initializes RUCB agent. 
        alpha -> "Exploration rate" similar to UCB.
        """
        super(RUCBAgent,self).__init__(n_arms)

        # UCB exp rate
        self.alpha = alpha

        # Keeps track of the best candidate
        self.best = None 

        # Time step
        self.time = 1

    def step(self):
        """(Override) Returns the pair that should be matched, using RUCB"""

        # Compute Upper Bounds for confidence intervals (UCB)
        total_matches = self.outcomes + np.transpose(self.outcomes)
        mask = (total_matches != 0) # This will prevent division by zero, setting 1 in those places instead.
        conf_interval_sizes = np.sqrt(self.alpha * np.where(mask, np.divide(np.log(self.time), total_matches, where=mask), 1))
        upper_bounds = np.where(mask, np.divide(self.outcomes, total_matches, where=mask), 1) + conf_interval_sizes
        np.fill_diagonal(upper_bounds, 1/2)

        # Select candidates to condorcet winner
        cond_winners = np.flatnonzero((upper_bounds >= 1/2).all(axis=1))

        # Select benchmarking arm
        a_c = None
        if cond_winners.size == 0:
            a_c = np.random.randint(0, self.n_arms)
            if self.best != a_c:
                self.best = None
        elif cond_winners.size == 1:
            a_c = cond_winners[0]
            self.best = a_c
        else:
            # Select with higher weight for the best one.
            weights = np.full(self.n_arms, 1/(2*(self.n_arms-1)) if self.best else 1/self.n_arms)
            if self.best:
                weights[self.best] = 1/2
            a_c = np.random.choice(self.n_arms, p=weights)

        # Select opponent as the tightest one with a_c
        score_vs_ac = upper_bounds[:, a_c]
        opponent_candidates = np.flatnonzero(score_vs_ac == score_vs_ac.max())
        if opponent_candidates.size == 1:
            a_d = opponent_candidates[0]
        else:
            # Remove a_c as the opponent, if neccessary
            opponent_candidates = np.delete(opponent_candidates, np.where(opponent_candidates == a_c))
            a_d = np.random.choice(opponent_candidates)

        # Increase time step
        self.time += 1

        return (a_c, a_d)
        
        
    def reset(self):
        """Fully resets the agent"""
        super().reset()
        self.time = 1
        self.best = None

    def get_name(self):
        return f"RUCB DB w/a: {self.alpha}"

"""
Copeland Confidence Bound (CCB) Dueling Bandit Agent.
Initially presented at https://arxiv.org/abs/1506.00312.
"""

import numpy as np
from .DBAgent import DBAgent

class CCBAgent(DBAgent):
    """
    Implements a dueling bandit agent following the CCB policy.
    """
    
    def __init__(self, n_arms, alpha=0.51):
        """
        Initializes CCB agent.

        Args:
            n_arms: number of arms
            alpha: "exploration rate" similar to UCB.
        """

        super(CCBAgent,self).__init__(n_arms)

        # UCB exp rate
        self.alpha = alpha

        # Keeps track of the best candidates
        self.best = set(range(n_arms))

        # Keeps track of the best opponents of each arm
        self.best_opponents = [set() for _ in range(n_arms)]

        # Estimates against how many arms does the Copeland Winner lose
        self.copeland_winner_losses = n_arms

        # Time step
        self.time = 1

    def step(self):
        """
        (Override) Returns the pair that should be matched, using CCB.

        Returns:
            Pair of indices (i,j) that the policy decided to pull.
        """

        # Compute Upper Bounds for confidence intervals (UCB) and lower bounds
        total_matches = self.outcomes + np.transpose(self.outcomes)
        mask = (total_matches != 0) # This will prevent division by zero, setting 1 in those places instead.
        conf_interval_sizes = np.sqrt(self.alpha * np.where(mask, np.divide(np.log(self.time), total_matches, where=mask), 1))
        upper_bounds = np.where(mask, np.divide(self.outcomes, total_matches, where=mask), 1) + conf_interval_sizes
        lower_bounds = np.where(mask, np.divide(self.outcomes, total_matches, where=mask), 1) - conf_interval_sizes
        np.fill_diagonal(upper_bounds, 1/2)
        np.fill_diagonal(lower_bounds, 1/2)

        # Compute upper and lower estimates for copeland scores
        cope_upper = np.count_nonzero((upper_bounds >= 1/2), axis=1) - 1
        cope_lower = np.count_nonzero((lower_bounds >= 1/2), axis=1) - 1

        # Compute copeland winner candidates for this round
        cope_winners = np.flatnonzero(cope_upper == cope_upper.max())
        
        # Reset disproven hypotheses
        for i in range(self.n_arms):
            for j in self.best_opponents[i]:
                if lower_bounds[i,j] > 0.5:
                    self.best = set(range(self.n_arms))
                    self.best_opponents = [set() for _ in range(self.n_arms)]
                    self.copeland_winner_losses = self.n_arms
        
        # Remove non-Copeland winners
        if self.best:
            copy = set(self.best)
            for i in copy:
                if cope_upper[i] < cope_lower[i]:
                    self.best.remove(i)
                    if len(self.best_opponents[i]) != self.copeland_winner_losses + 1:
                        self.best_opponents[i] = set(np.flatnonzero((upper_bounds[i,:] < 1/2)))
        else:
            # Reset hypotheses
            self.best = set(range(self.n_arms))
            self.best_opponents = [set() for _ in range(self.n_arms)]
            self.copeland_winner_losses = self.n_arms

        # Add Copeland winners
        for i in cope_winners:
            if cope_lower[i] == cope_upper[i]:
                self.best.add(i)
                self.best_opponents[i] = set()
                self.copeland_winner_losses = self.n_arms - 1 - cope_upper[i]
                for j in range(self.n_arms):
                    if i == j:
                        continue
                    if len(self.best_opponents[j]) < self.copeland_winner_losses + 1:
                        self.best_opponents[j] = set()
                    elif len(self.best_opponents[j]) > self.copeland_winner_losses + 1:
                        self.best_opponents[j] = set(np.random.choice(list(self.best_opponents[j]), 
                                                     size=self.copeland_winner_losses+1, replace=False))

        # Increase time step
        self.time += 1

        # Probability of 1/4 of using best_opponents
        if np.random.random() < 1/4:
            pairs = [(i,j) for i in range(self.n_arms) for j in range(self.n_arms) if j in self.best_opponents[i] and lower_bounds[i,j] <= 1/2 and upper_bounds[i,j] <= 1/2]
            if pairs:
                return pairs[np.random.randint(0,len(pairs))]

        # Probability of 2/3 of limiting current bests to overall bests
        if np.random.random() < 2/3:
            intersected = self.best.intersection(cope_winners)
            if intersected:
                cope_winners = np.array(list(intersected))

        a_c = np.random.choice(cope_winners)

        # Select opponent as the tightest one with a_c, probability 1/2 of only using best_opponents
        score_vs_ac = upper_bounds[:, a_c]
        if np.random.random() < 1/2:
            to_discard = set(range(self.n_arms)).difference(self.best_opponents[a_c])
            score_vs_ac[list(to_discard)] = np.NINF
        opponent_candidates = np.flatnonzero(score_vs_ac == score_vs_ac.max())
        # Remove depending on lower bound
        np.delete(opponent_candidates, np.where(lower_bounds[opponent_candidates, a_c] > 0.5))
        if opponent_candidates.size == 1:
            a_d = opponent_candidates[0]
        else:
            # Remove a_c as the opponent, if neccessary
            opponent_candidates = np.delete(opponent_candidates, np.where(opponent_candidates == a_c))
            a_d = np.random.choice(opponent_candidates)

        return (a_c, a_d)
        
        
    def reset(self):
        """
        Fully resets the agent
        """

        super().reset()
        self.time = 1
        self.best = set(range(self.n_arms))
        self.best_opponents = [set() for _ in range(self.n_arms)]
        self.copeland_winner_losses = self.n_arms
        
    def get_name(self):
        """
        String representation of the agent.

        Returns:
            string representing the agent.
        """
        return f"CCB DB w/alpha: {self.alpha}"

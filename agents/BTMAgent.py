"""
Beat The Mean (BTM) dueling bandit agent.
"""

import random
import numpy as np
from numpy.core.numeric import Inf
from .DBAgent import DBAgent

class BTMAgent(DBAgent):
    
    def __init__(self, n_arms, horizon, gamma=1):
        """
        Initializes IF Agent. The parameter "horizon" indicates the time horizon
        for the algorithm to run. The parameter gamma represents transitivity relaxation.
        """
        super(BTMAgent,self).__init__(n_arms)
        self.horizon = horizon
        self.gamma = gamma

        # 1 - delta = Confidence required to conclude winner. 
        # The more horizon, the more confidence in the selected winner.
        self.delta = 1/(2*n_arms*horizon)

        # Current working set.
        self.working_set = list(range(n_arms))

        # Represents win counter (Wb). Position i,j means i beats j.
        self.wins = np.zeros((self.n_arms, self.n_arms))

        # Represents comparison counter (Nb). Position i,j means i was compared to j.
        self.comparisons = np.zeros((self.n_arms, self.n_arms))

        # Represents probabilities of beating the mean bandit
        self.probs = np.array([1/2] * self.n_arms)

        # Counter of steps
        self.steps = 0

    def reward(self, n_arm_1, n_arm_2, one_wins):
        """
        Updates the knowledge given the reward. Since it's a Dueling Bandit, the reward
        is a boolean indicating whether the first arm wins or not.
        """
        # If winner was already selected no need to update
        if len(self.working_set) == 1 or self.steps >= self.horizon:
            return

        # Update wins and comparisons of the chosen arm against "the mean".
        if one_wins:
            self.wins[n_arm_1][n_arm_2] += 1
        self.comparisons[n_arm_1][n_arm_2] += 1

        wins_per_arm = np.sum(self.wins,axis=1)
        comps_per_arm = np.sum(self.comparisons,axis=1)
        self.probs[n_arm_1] = wins_per_arm[n_arm_1] / comps_per_arm[n_arm_1]

        # Increase step counter
        self.steps += 1

        # Now we check if updates to the working set are needed.

        # Get the minimum number of comparisons from within the working set
        comp_min = np.min(comps_per_arm[self.working_set])
        # Get confidence interval
        conf = self.gamma**2 * np.sqrt((1/comp_min) * np.log(1/self.delta)) if comp_min != 0 else 1

        if(self.gamma > 1):
            conf *= 3

        worst_prob = np.min(self.probs[self.working_set])
        best_prob = np.max(self.probs[self.working_set])

        # Update working set removing the loser, if any
        if worst_prob + conf <= best_prob - conf:
            
            losers_all = np.flatnonzero(self.probs == worst_prob)
            losers = losers_all[np.in1d(losers_all, self.working_set, assume_unique=True)] # Only Working Set
            loser = np.random.choice(losers)

            # Remove every comparison and win towards the loser (i.e, raise the mean)
            self.wins[:,loser] = np.zeros(self.n_arms)
            self.comparisons[:,loser] = np.zeros(self.n_arms)

            self.working_set.remove(loser)

            # Recompute the probabilities
            for arm in range(self.n_arms):
                if arm in self.working_set:
                    wins_per_arm = np.sum(self.wins,axis=1)
                    comps_per_arm = np.sum(self.comparisons,axis=1)
                    self.probs[arm] = wins_per_arm[arm] / comps_per_arm[arm]
                else:
                    self.probs[arm] = 0


    def step(self):
        """(Override) Returns the pair that should be matched, using BTM"""

        # If we're finished determining a leader, we stick to it.
        if len(self.working_set) == 1 or self.steps >= self.horizon:
            best = np.argmax(self.probs)
            return best, best

        # Otherwise, we select the arms according to BTM

        # Get the less compared arm from working set, with random tie breaking
        comps_per_arm = np.sum(self.comparisons,axis=1)
        comps_min = np.min(comps_per_arm[self.working_set])
        least_comps_all = np.flatnonzero(comps_per_arm == comps_min) # Might contain indices not in WS
        least_comps = least_comps_all[np.in1d(least_comps_all, self.working_set, assume_unique=True)] # Ensure WS
        arm1 = np.random.choice(least_comps)
        # Choose another arm at random
        arm2 = np.random.choice(self.working_set)

        return arm1, arm2
        
    def reset(self):
        """Fully resets the agent"""
        super().reset()
        self.working_set = list(range(self.n_arms))
        self.wins = np.zeros((self.n_arms, self.n_arms))
        self.comparisons = np.zeros((self.n_arms, self.n_arms))
        self.probs = np.array([1/2] * self.n_arms)
        self.steps = 0

    def get_name(self):
        return f"Beat the Mean DB w/horizon={self.horizon}, gamma={self.gamma}"

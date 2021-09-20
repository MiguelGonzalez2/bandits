"""
Interleaved Filter (IF) dueling bandit agent.
"""

import random
import numpy as np
from .DBAgent import DBAgent

class IFAgent(DBAgent):
    
    def __init__(self, n_arms, horizon):
        """
        Initializes IF Agent. The parameter "horizon" indicates the time horizon
        for the algorithm to run.
        """
        super(IFAgent,self).__init__(n_arms)
        self.horizon = horizon

        # 1 - delta = Confidence required to conclude winner. 
        # The more horizon, the more confidence in the selected winner.
        self.delta = 1/(horizon * (n_arms**2))

        # Current best candidate
        self.leader = 0

        # Candidates that are eligible to face the current best
        self.candidates = range(1, n_arms)

        # Index of the candidate that should face the leader next.
        self.turn = 0

    def reward(self, n_arm_1, n_arm_2, one_wins):
        """
        Updates the knowledge given the reward. Since it's a Dueling Bandit, the reward
        is a boolean indicating whether the first arm wins or not.
        In Interleaved Filter, it updates the estimates and confidence intervals.
        """

        # If the leader has already been selected, no need to update.
        if not self.candidates:
            return

        super().reward(n_arm_1, n_arm_2, one_wins)

        # Advance the turn
        self.turn += 1

        # If there are candidates remaining, proceed
        if self.turn < len(self.candidates):
            return

        # Otherwise, we update statistics and candidates.

        # We keep track of losing and winning candidates
        soft_losers = [] # Candidates that are worse, but not confidently
        hard_losers = [] # Candidates that are worse confidently
        winner = None # Candidate that is better, confidently.

        for candidate in self.candidates:

            # Probability that the candidate beats the leader
            prob = self.get_ratio(candidate, self.leader)

            # Size of the confidence interval for this match
            conf = np.sqrt(4*np.log(1/self.delta)/self.get_comparison_count(candidate, self.leader))

            # If the leader confidently beats the candidate, remove it
            if prob + conf < 1/2:
                hard_losers.append(candidate)

            # If the leader beats the candidate, mark it for possible pruning
            if prob < 1/2:
                soft_losers.append(candidate)

            # If candidate confidently beats the leader, mark it as winner.
            if winner == None and prob - conf > 1/2:
                winner = candidate

        # Remove confident losers
        self.candidates = list(set(self.candidates) - set(hard_losers))

        if winner != None:

            # Pruning
            self.candidates = list(set(self.candidates) - set(soft_losers))

            # New leader
            self.leader = winner
            self.candidates = self.candidates.remove(winner)

            # Reset the probabilities of the previous round
            super().reset()

        # Reset turn
        self.turn = 0

    def step(self):
        """(Override) Returns the pair that should be matched, using Interleaved Filter"""

        # If we're finished determining a leader, we stick to it.
        if not self.candidates:
            return self.leader, self.leader

        return self.leader, self.candidates[self.turn]
        
    def reset(self):
        """Fully resets the agent"""
        super().reset()
        self.leader = 0
        self.candidates = range(1, self.n_arms)
        self.turn = 0

    def get_name(self):
        return f"Interleaved Filter DB w/horizon={self.horizon}"

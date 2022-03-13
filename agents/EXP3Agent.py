"""
EXP3 MAB agent.
Introduced in https://cseweb.ucsd.edu/~yfreund/papers/bandits.pdf.
"""

import numpy as np
from .MABAgent import MABAgent

def EXP3Gamma(best_value, n_rounds, n_arms):
    """
    Helper function to obtain "theoretically good" value for the exploration rate.

    Args:
        best_value: upper bound for the reward values
        n_rounds: number of iterations for EXP3
        n_arms: number of arms

    Returns:
        value that should be given to EXP3's exploration rate.
    """
    g = best_value * n_rounds # Upper bound for final reward
    return np.min(np.array([1,np.sqrt(n_arms*np.log(n_arms)/(np.exp(1) * g))]))

class EXP3Agent(MABAgent):
    """
    Implements a multi armed bandit agent following the EXP3 policy.
    """ 

    def __init__(self, n_arms, exploration_rate = 0.1, optimism=None):
        """
        Initializes EXP3 Agent. 
        
        Args:
            n_arms: number of arms
            exploration_rate: weight (0 to 1) that is given to exploration vs exploitation.
            optimism: starting estimation for the value of every arm.
        """
        super(EXP3Agent,self).__init__(n_arms, optimism)
        self.exprate = exploration_rate
        self.optimism = optimism
        self.weights = np.ones(n_arms) # EXP3 weights
        self.probs = np.empty(n_arms) # EXP3 probabilities

    def step(self):
        """
        (Override) Returns the arm that should be pulled, using EXP3.

        Returns:
            Index i of the arm that the policy decided to pull.
        """

        # Compute each arms probabilities
        self.probs = (1-self.exprate)*(self.weights/sum(self.weights)) + self.exprate/self.n_arms

        return np.random.choice(range(self.n_arms), p=self.probs)

    def reward(self, n_arm, reward):
        """
        Updates the knowledge given the reward. 

        Args:
            n_arm: pulled arm.
            reward: numerical reward obtained.
        """
        super().reward(n_arm, reward)

        #Update weights
        self.weights[n_arm] = self.weights[n_arm] * np.exp(self.exprate * reward / (self.n_arms * self.probs[n_arm]))

    def reset(self):
        """
        Fully resets the agent
        """
        super().reset()
        self.weights = np.ones(self.n_arms)
        self.probs = np.empty(self.n_arms)


    def get_name(self):
        """
        String representation of the agent.

        Returns:
            string representing the agent.
        """
        return f"EXP3, gamma={self.exprate}"

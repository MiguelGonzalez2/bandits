"""
Contains and updates MAB metrics
"""

import numpy as np

def new_average(old_average, new_value, value_count):
    """
    Helper function to update average value.
    old_average -> old value
    new_value -> new_value
    value_count -> total value count including new value
    """
    return 1/(value_count) * ((value_count-1)*old_average + new_value)

class MABMetrics():

    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.rewards = np.empty(n_epochs)
        self.sum_rewards = 0 
        self.regrets = np.empty(n_epochs)
        self.value_counts = np.zeros(n_epochs)

    def update(self, epoch, reward, optimal_reward):

        # Get how many data we have for that given epoch
        self.value_counts[epoch] += 1
        n_values = self.value_counts[epoch]
        
        # Update

        self.rewards[epoch] = new_average(self.rewards[epoch], reward, n_values)

        self.sum_rewards += reward
        new_regret = (epoch+1)*optimal_reward - self.sum_rewards
        self.regrets[epoch] = new_average(self.regrets[epoch], new_regret, n_values) 

    def new_iteration(self):
        """
        Call if a new iteration (aka different environment) has begun
        """
        self.sum_rewards = 0

    def get_metrics(self):
        return {'reward': self.rewards, 'regret': self.regrets}

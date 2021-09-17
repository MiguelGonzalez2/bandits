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

class Metrics():

    def __init__(self, n_epochs):
        """
        Initializes the metrics object.
        """
        self.n_epochs = n_epochs

        # Average reward obtained, only used for MABs
        self.rewards = np.empty(n_epochs)

        # In the case of dueling bandits, strong regret measures
        # the worst "classical" regret out of the two dueling bandits,
        # and weak regret measures the best "classical" regret.
        # In the case of MABs they will match to the actual regret.
        self.strong_regrets = np.empty(n_epochs) 
        self.weak_regrets = np.empty(n_epochs)

        self.value_counts = np.zeros(n_epochs)
        self.sum_weak_rewards = 0 
        self.sum_strong_rewards = 0 

    def update_dueling(self, epoch, reward1, reward2, optimal_reward):
        """
        Updates the data after a dueling bandits pull, that is, after
        a comparison between two arms has been carried out.
        """
        # Get how many data we have for that given epoch
        self.value_counts[epoch] += 1
        n_values = self.value_counts[epoch]
        
        # Get the worst reward in reward1
        if reward1 > reward2:
            reward1, reward2 = reward2, reward1

        # This should not matter for Dueling Bandits, however, we update it to the best reward
        # for consistency.
        self.rewards[epoch] = new_average(self.rewards[epoch], reward2, n_values)

        # Update regrets
        self.sum_weak_rewards += reward2
        self.sum_strong_rewards += reward1

        # Weak regret
        new_regret = (epoch+1)*optimal_reward - self.sum_weak_rewards
        self.weak_regrets[epoch] = new_average(self.weak_regrets[epoch], new_regret, n_values) 

        # Strong regret
        new_regret = (epoch+1)*optimal_reward - self.sum_strong_rewards
        self.strong_regrets[epoch] = new_average(self.strong_regrets[epoch], new_regret, n_values) 

    def update(self, epoch, reward, optimal_reward):
        """
        Updates the metrics using a single reward (that is, for the case of MABs)
        """
        self.update_dueling(epoch, reward, reward, optimal_reward)
        
    def new_iteration(self):
        """
        Call if a new iteration (aka different environment) has begun
        """
        self.sum_weak_rewards = 0
        self.sum_strong_rewards = 0

    def get_metrics(self):
        return {'reward': self.rewards,
                'regret': self.weak_regrets,
                'weak_regret': self.weak_regrets,
                'strong_regret': self.strong_regrets}

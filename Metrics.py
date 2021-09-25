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
        self.regrets = np.empty(n_epochs)
        self.strong_regrets = np.empty(n_epochs) 
        self.weak_regrets = np.empty(n_epochs)

        # Times that the strategy chose the optimal reward
        self.chose_optimal = np.empty(n_epochs)

        self.value_counts = np.zeros(n_epochs)
        self.sum_rewards = 0
        self.sum_weak_rewards = 0 
        self.sum_strong_rewards = 0 

    def update_dueling(self, epoch, environment, arm1, arm2, reward1, reward2, optimal_arm, optimal_reward):
        """
        Updates the data after a dueling bandits pull, that is, after
        a comparison between two arms has been carried out.
        """
        # Get how many data we have for that given epoch
        self.value_counts[epoch] += 1
        n_values = self.value_counts[epoch]
        
        # Get the worst arm in position 1
        if environment.arms[arm1] > environment.arms[arm2]:
            reward1, reward2 = reward2, reward1
            arm1, arm2 = arm2, arm1

        # This should not matter for Dueling Bandits, however, we update it to the best reward
        # for consistency.
        self.rewards[epoch] = new_average(self.rewards[epoch], reward2, n_values)

        # Get the probability that optimal arm beats each arm. (We only need displacement from 1/2) 
        prob1 = environment.get_probability_dueling(optimal_arm, arm1) - 1/2
        prob2 = environment.get_probability_dueling(optimal_arm, arm2) - 1/2

        # Update regrets
        self.sum_rewards += reward2 # We choose the supposedly better reward here.
        self.sum_weak_rewards += prob2
        self.sum_strong_rewards += prob1

        # Standard MAB regret
        new_regret = (epoch+1)*optimal_reward - self.sum_rewards
        self.regrets[epoch] = new_average(self.regrets[epoch], new_regret, n_values) 

        # Weak DB regret
        self.weak_regrets[epoch] = new_average(self.weak_regrets[epoch], self.sum_weak_rewards, n_values) 

        # Strong DB regret
        self.strong_regrets[epoch] = new_average(self.strong_regrets[epoch], self.sum_strong_rewards, n_values) 

        # Optimal reward
        self.chose_optimal[epoch] = new_average(self.chose_optimal[epoch], int(arm2 == optimal_arm), n_values)
        

    def update(self, epoch, environment, arm, reward, optimal_arm, optimal_reward):
        """
        Updates the metrics using a single reward (that is, for the case of MABs)
        """
        self.update_dueling(epoch, environment, arm, arm, reward, reward, optimal_arm, optimal_reward)
        
    def new_iteration(self):
        """
        Call if a new iteration (aka different environment) has begun
        """
        self.sum_rewards = 0
        self.sum_weak_rewards = 0
        self.sum_strong_rewards = 0

    def get_metrics(self):
        return {'reward': self.rewards,
                'regret': self.regrets,
                'weak_regret': self.weak_regrets,
                'strong_regret': self.strong_regrets,
                'optimal_percent': self.chose_optimal}

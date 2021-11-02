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
        self.rewards = np.zeros(n_epochs)

        # In the case of dueling bandits, strong regret measures
        # the worst "classical" regret out of the two dueling bandits,
        # and weak regret measures the best "classical" regret.
        # In the case of MABs they will match to the actual regret.
        self.regrets = np.zeros(n_epochs)
        self.strong_regrets = np.zeros(n_epochs) 
        self.weak_regrets = np.zeros(n_epochs)

        # Copeland regret for non-condorcet situations
        self.copeland_regrets = np.zeros(n_epochs)
        self.copeland_regrets_non_cumulative = np.zeros(n_epochs)

        # Times that the strategy chose the optimal reward
        self.chose_optimal = np.zeros(n_epochs)

        self.value_counts = np.zeros(n_epochs)
        self.sum_rewards = 0
        self.sum_weak_rewards = 0 
        self.sum_strong_rewards = 0 
        self.sum_copeland_rewards = 0

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

        # We consider the reward the average of the rewards of each individual choice.
        # This will punish bad user experiences due to presenting poor results.
        self.rewards[epoch] = new_average(self.rewards[epoch], (reward1 + reward2) / 2, n_values)

        # Get the probability that optimal arm beats each arm. (We only need displacement from 1/2) 
        prob1 = environment.get_probability_dueling_cached(optimal_arm, arm1) - 1/2
        prob2 = environment.get_probability_dueling_cached(optimal_arm, arm2) - 1/2

        # Get the copeland individual regret (similar to before, but compare against copeland winners)
        cop_score1 = environment.get_copeland_regret(arm1)
        cop_score2 = environment.get_copeland_regret(arm2)

        # Update regrets
        self.sum_rewards += np.clip((reward1 + reward2) / 2, 0, None)
        self.sum_weak_rewards += np.clip(prob2, 0, None)
        self.sum_strong_rewards += np.clip(prob1, 0, None)
        self.sum_copeland_rewards += np.clip((cop_score1 + cop_score2) / 2, 0, None)

        # Standard MAB regret
        new_regret = (epoch+1)*optimal_reward - self.sum_rewards
        self.regrets[epoch] = new_average(self.regrets[epoch], new_regret, n_values) 

        # Weak DB regret
        self.weak_regrets[epoch] = new_average(self.weak_regrets[epoch], self.sum_weak_rewards, n_values) 

        # Strong DB regret
        self.strong_regrets[epoch] = new_average(self.strong_regrets[epoch], self.sum_strong_rewards, n_values) 

        # Copeland DB regret
        self.copeland_regrets[epoch] = new_average(self.copeland_regrets[epoch], self.sum_copeland_rewards, n_values) 
        self.copeland_regrets_non_cumulative[epoch] = new_average(self.copeland_regrets_non_cumulative[epoch], (cop_score1 + cop_score2) / 2, n_values)

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
        self.sum_copeland_rewards = 0

    def get_metrics(self):
        return {'reward': self.rewards,
                'regret': self.regrets,
                'dueling_regret': self.copeland_regrets,
                'copeland_regret': self.copeland_regrets,
                'dueling_regret_non_cumulative': self.copeland_regrets_non_cumulative,
                'copeland_regret_non_cumulative': self.copeland_regrets_non_cumulative,
                'weak_regret': self.weak_regrets,
                'strong_regret': self.strong_regrets,
                'optimal_percent': self.chose_optimal}

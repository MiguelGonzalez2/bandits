"""
MABSimulation Environment
"""

# TODO Run multiple arm-types at once
# TODO Average over many environments, percentage of optimal
# TODO Add metrics

import numpy as np
from agents import EpsilonGreedyAgent
from environments import GaussianEnvironment
import matplotlib.pyplot as plt

class MABSimulation():
    """Class that carries MAB simulations""" 

    def __init__(self, agent, environment, n_epochs):
        self.agent = agent
        self.environment = environment
        self.n_epochs = n_epochs
        self.rewards = np.empty(n_epochs)
        self.sum_rewards = 0 
        self.regrets = np.empty(n_epochs)
        self.optimal_percents = np.empty(n_epochs)
        self.sum_optimals = 0

    def run(self):
        optimal = self.environment.get_optimal()
        optimal_value = self.environment.get_optimal_value()
        for i in range(self.n_epochs):
            # Ask the agent for an action
            arm = self.agent.step()
            # Get reward
            reward = self.environment.step(arm)
            # Feed agent
            self.agent.reward(arm, reward)
            # Update metrics
            self.rewards[i] = reward
            self.sum_rewards += reward
            self.regrets[i] = (i+1)*optimal_value - self.sum_rewards 
            if arm == optimal:
                self.sum_optimals += 1
            self.optimal_percents[i] = self.sum_optimals/(i+1)

    def plot_rewards(self):
        return plt.plot(self.rewards)

    def plot_regret(self):
        return plt.plot(self.regrets)

    def plot_optimals(self):
        return plt.plot(self.optimal_percents)

### Test
n_arms = 10
agent = EpsilonGreedyAgent.EpsilonGreedyAgent(n_arms,0.01)
environment = GaussianEnvironment.GaussianEnvironment(n_arms)
sim = MABSimulation(agent, environment, 2000)
sim.run()
sim.plot_optimals()
plt.show()


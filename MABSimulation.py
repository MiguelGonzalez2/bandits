"""
MABSimulation Environment
"""

# TODO work on plot labels

import MABMetrics as mm
from agents import EpsilonGreedyAgent, UCBAgent, EXP3Agent
from environments import GaussianEnvironment
import matplotlib.pyplot as plt

class MABSimulation():
    """Class that carries MAB simulations""" 

    def __init__(self, agents, environment, n_epochs, n_repeats=1):
        """
        Agents: list of agents to simulate
        Environment: Environment object with the arms
        n_epochs: Nº of iterations per agent on a given environment
        n_repeats: Nº of environments per agent for robustness
        """
        self.agents = agents
        self.environment = environment
        self.n_epochs = n_epochs
        self.metrics = [mm.MABMetrics(n_epochs) for i in range(len(agents))]
        self.n_repeats = n_repeats

    def run(self):

        # Loops through the several environments
        for iteration in range(self.n_repeats):

            optimal_value = self.environment.get_optimal_value()

            # Loops through the several agents
            for agent_id, agent in enumerate(self.agents):

                # Carry one experiment
                for i in range(self.n_epochs):
                    # Ask the agent for an action
                    arm = agent.step()
                    # Get reward
                    reward = self.environment.step(arm)
                    # Feed agent
                    agent.reward(arm, reward)
                    # Update metrics
                    self.metrics[agent_id].update(i, reward, optimal_value)
                
                self.environment.soft_reset()
                self.metrics[agent_id].new_iteration()
                agent.reset()

            self.environment.reset()

    def plot_metrics(self, metric_name):
        plots = []
        for i in range(len(self.agents)):
            plots += plt.plot(self.metrics[i].get_metrics()[metric_name], label=self.agents[i].get_name())
        plt.legend(plots, [self.agents[i].get_name() for i in range(len(self.agents))])
        plt.show()

### Test
n_arms = 10
n_iterations = 2000
n_simulations = 2000
agent1 = EpsilonGreedyAgent.EpsilonGreedyAgent(n_arms,0.1)
agent2 = EpsilonGreedyAgent.EpsilonGreedyAgent(n_arms,0)
agent3 = UCBAgent.UCBAgent(n_arms)
agent4 = EXP3Agent.EXP3Agent(n_arms, exploration_rate=EXP3Agent.EXP3Gamma(3, n_iterations, n_arms))
environment = GaussianEnvironment.GaussianEnvironment(n_arms)
sim = MABSimulation([agent1, agent2, agent3, agent4], environment, n_iterations, n_simulations)
sim.run()
sim.plot_metrics('reward')


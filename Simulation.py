"""
Simulation Environment
"""

from agents.MultiSBMAgent import MultiSBMAgent
import Metrics as mm
from agents.EpsilonGreedyAgent import EpsilonGreedyAgent
from agents.UCBAgent import UCBAgent
from agents.EXP3Agent import EXP3Agent, EXP3Gamma
from agents.ThompsonBetaAgent import ThompsonBetaAgent
from agents.IFAgent import IFAgent
from agents.BTMAgent import BTMAgent
from agents.DoublerAgent import DoublerAgent
from agents.SparringAgent import SparringAgent
from environments import GaussianEnvironment, BernoulliEnvironment
import matplotlib.pyplot as plt
import numpy as np

class Simulation():
    """Class that carries MAB and DB simulations""" 

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
        self.metrics = [mm.Metrics(n_epochs) for i in range(len(agents))]
        self.n_repeats = n_repeats

    def run(self):

        # Loops through the several environments
        for iteration in range(self.n_repeats):

            optimal_arm = self.environment.get_optimal()
            optimal_value = self.environment.get_optimal_value()

            # Loops through the several agents
            for agent_id, agent in enumerate(self.agents):

                # Carry one experiment
                for i in range(self.n_epochs):
                    # MAB's case:
                    if not agent.is_dueling:
                        # Ask the agent for an action
                        arm = agent.step()
                        # Get reward
                        reward = self.environment.step(arm)
                        # Feed agent
                        agent.reward(arm, reward)
                        # Update metrics
                        self.metrics[agent_id].update(i, self.environment, arm, reward, optimal_arm, optimal_value)
                    # DB's case:
                    else:
                        # Ask the agent for an action
                        arm1, arm2 = agent.step()
                        # Get rewards in order to compare
                        reward1, reward2 = self.environment.dueling_step(arm1, arm2)
                        # Feed agent with the result of the comparison only
                        agent.reward(arm1, arm2, reward1 > reward2)
                        # Update metrics
                        self.metrics[agent_id].update_dueling(i, self.environment, arm1, arm2, reward1, reward2, optimal_arm, optimal_value)
                
                self.environment.soft_reset()
                self.metrics[agent_id].new_iteration()
                agent.reset()

            self.environment.reset()

    def plot_metrics(self, metric_name):
        plots = []
        for i in range(len(self.agents)):
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plots += plt.plot(self.metrics[i].get_metrics()[metric_name], label=self.agents[i].get_name())
        plt.legend(plots, [self.agents[i].get_name() for i in range(len(self.agents))])
        plt.show()

### Test
n_arms = 10
n_iterations = 2000
n_simulations = 100
agents = []
agents.append(EpsilonGreedyAgent(n_arms,0.1))
agents.append(EpsilonGreedyAgent(n_arms,0))
agents.append(UCBAgent(n_arms))
agents.append(EXP3Agent(n_arms, exploration_rate=EXP3Gamma(3, n_iterations, n_arms)))
agents.append(ThompsonBetaAgent(n_arms))
agents.append(IFAgent(n_arms, n_iterations*0.9))
agents.append(BTMAgent(n_arms, n_iterations*0.9, 1))
#environment = BernoulliEnvironment.BernoulliEnvironment(n_arms)
environment = GaussianEnvironment.GaussianEnvironment(n_arms)

reductors = []
reductors.append(IFAgent(n_arms, n_iterations*0.2))
reductors.append(BTMAgent(n_arms, n_iterations*0.2))
reductors.append(DoublerAgent(n_arms, UCBAgent(n_arms)))
reductors.append(MultiSBMAgent(n_arms, UCBAgent, [n_arms]))
reductors.append(SparringAgent(n_arms, UCBAgent(n_arms), UCBAgent(n_arms)))
#reductors.append(DoublerAgent(n_arms, ThompsonBetaAgent(n_arms)))
#reductors.append(MultiSBMAgent(n_arms, ThompsonBetaAgent, [n_arms]))
reductors.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms), ThompsonBetaAgent(n_arms)))
sim = Simulation(reductors, environment, n_iterations, n_simulations)
sim.run()
sim.plot_metrics('optimal_percent')
sim.plot_metrics('regret')
sim.plot_metrics('reward')
sim.plot_metrics('weak_regret')
sim.plot_metrics('strong_regret')


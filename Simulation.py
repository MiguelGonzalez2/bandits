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
from agents.DTSAgent import DTSAgent
from agents.RUCBAgent import RUCBAgent
from agents.CCBAgent import CCBAgent
from environments import GaussianEnvironment, BernoulliEnvironment, CyclicRPSEnvironment, NoisyGaussianEnvironment
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

class Simulation():
    """Class that carries MAB and DB simulations""" 

    def __init__(self, agents, environment, n_epochs, n_repeats=1, sample_frequency=1):
        """
        Agents: list of agents to simulate
        Environment: Environment object with the arms
        n_epochs: Nº of iterations per agent on a given environment
        n_repeats: Nº of environments per agent for robustness
        sample_frequency: Metrics are taken after this amount of steps.
        """
        self.agents = agents
        self.environment = environment
        self.n_epochs = n_epochs
        self.metrics = [mm.Metrics(n_epochs) for i in range(len(agents))]
        self.n_repeats = n_repeats
        self.sample_frequency = sample_frequency

    def run(self):

        # Loops through the several environments
        for iteration in tqdm(range(self.n_repeats)):

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
                        if i % self.sample_frequency == 0:
                            self.metrics[agent_id].update(i, self.environment, arm, reward, optimal_arm, optimal_value)
                    # DB's case:
                    else:
                        # Ask the agent for an action
                        arm1, arm2 = agent.step()
                        # Get rewards in order to compare
                        reward1, reward2 = self.environment.dueling_step(arm1, arm2)
                        # Feed agent with the result of the comparison only. Ties are broken randomly
                        agent.reward(arm1, arm2, reward1 > reward2 if reward1 != reward2 else random.choice([True, False]))
                        # Update metrics
                        if i % self.sample_frequency == 0:
                            self.metrics[agent_id].update_dueling(i, self.environment, arm1, arm2, reward1, reward2, optimal_arm, optimal_value)
                
                self.environment.soft_reset()
                self.metrics[agent_id].new_iteration()
                agent.reset()

            self.environment.reset()

    def plot_metrics(self, metric_name, scale="linear"):
        plots = []
        x = range(0, self.n_epochs, self.sample_frequency)
        for i in range(len(self.agents)):
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plots += plt.plot(x, self.metrics[i].get_metrics()[metric_name][x], label=self.agents[i].get_name())
        plt.legend(plots, [self.agents[i].get_name() for i in range(len(self.agents))])
        plt.xscale(scale)
        plt.title(f"{self.agents[0].n_arms} arms, {self.n_repeats} simulations with {self.n_epochs} epochs each. Environment: {self.environment.get_name()}")
        plt.show()

### Test
n_arms = 5
n_iterations = 500000
n_simulations = 50
agents = []
agents.append(EpsilonGreedyAgent(n_arms,0.1))
agents.append(EpsilonGreedyAgent(n_arms,0))
agents.append(UCBAgent(n_arms))
agents.append(EXP3Agent(n_arms, exploration_rate=EXP3Gamma(3, n_iterations, n_arms)))
agents.append(ThompsonBetaAgent(n_arms))
agents.append(IFAgent(n_arms, n_iterations*0.8))
agents.append(BTMAgent(n_arms, n_iterations*0.8, 1))
# environment = BernoulliEnvironment.BernoulliEnvironment(n_arms)
# environment = GaussianEnvironment.GaussianEnvironment(n_arms)
# environment = CyclicRPSEnvironment.CyclicRPSEnvironment(n_arms, std=0.2)
environment = NoisyGaussianEnvironment.NoisyGaussianEnvironment(n_arms, d=0.5)

reductors = []
#reductors.append(IFAgent(n_arms, n_iterations))
reductors.append(BTMAgent(n_arms, n_iterations))
#reductors.append(DoublerAgent(n_arms, UCBAgent(n_arms)))
#reductors.append(MultiSBMAgent(n_arms, UCBAgent, [n_arms]))
#reductors.append(SparringAgent(n_arms, UCBAgent(n_arms), UCBAgent(n_arms)))
#reductors.append(DoublerAgent(n_arms, ThompsonBetaAgent(n_arms)))
#reductors.append(MultiSBMAgent(n_arms, ThompsonBetaAgent, [n_arms]))
reductors.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms), ThompsonBetaAgent(n_arms)))
reductors.append(DTSAgent(n_arms))
reductors.append(RUCBAgent(n_arms))
reductors.append(CCBAgent(n_arms))
sim = Simulation(reductors, environment, n_iterations, n_simulations, sample_frequency=1000)
sim.run()
#sim.plot_metrics('optimal_percent')
#sim.plot_metrics('regret')
sim.plot_metrics('copeland_regret', scale='log')
sim.plot_metrics('copeland_regret_non_cumulative', scale='log')
#sim.plot_metrics('reward')
#sim.plot_metrics('weak_regret')
#sim.plot_metrics('strong_regret')


"""
Simulation for arm number impact in DB.
"""

from simulation.Simulation import Simulation
from simulation.Experiment import Experiment
from agents.MultiSBMAgent import MultiSBMAgent
from agents.ThompsonBetaAgent import ThompsonBetaAgent
from agents.UCBAgent import UCBAgent
from agents.IFAgent import IFAgent
from agents.BTMAgent import BTMAgent
from agents.DoublerAgent import DoublerAgent
from agents.SparringAgent import SparringAgent
from agents.DTSAgent import DTSAgent
from agents.EXP3Agent import EXP3Agent, EXP3Gamma
from agents.EpsilonGreedyAgent import EpsilonGreedyAgent
from agents.RUCBAgent import RUCBAgent
from agents.CCBAgent import CCBAgent
from agents.RandomAgent import RandomAgent
from environments import GaussianEnvironment

N_EPOCHS = 10000
N_REPEATS = 20 # Number of repeats for the largest amount of arms. The remaining arms are run more times since its cheaper.
MAX_N_REPEATS = 200 # Maximum number of repeats

N_ARM_VALUES = [10, 20, 30, 50, 100, 150, 200]

sim = Simulation(f"MAB change arm number, {N_EPOCHS} epochs, equally spaced arms")

for n_arms in N_ARM_VALUES:
    agents = []
    agents.append(RandomAgent(n_arms))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms, alpha_zero=1, beta_zero=10), ThompsonBetaAgent(n_arms, alpha_zero=1, beta_zero=10)))
    agents.append(SparringAgent(n_arms, UCBAgent(n_arms=n_arms, exploration_rate=0.1), UCBAgent(n_arms=n_arms, exploration_rate=0.1)))
    agents.append(DTSAgent(n_arms, alpha=0.05, beta=0.01, gamma=1))
    agents.append(ThompsonBetaAgent(n_arms, alpha_zero=1, beta_zero=10, failure_thres=n_arms/2))
    agents.append(UCBAgent(n_arms=n_arms, exploration_rate=0.1))
    agents.append(EXP3Agent(n_arms=n_arms,exploration_rate=EXP3Gamma(n_arms+1,N_EPOCHS,n_arms=n_arms)))
    agents.append(EpsilonGreedyAgent(n_arms=n_arms))

    environ = GaussianEnvironment.GaussianEnvironment(n_arms, values = list(range(n_arms)))

    n_repeats = min(int(N_REPEATS * float(N_ARM_VALUES[-1])**2 / float(n_arms)**2), MAX_N_REPEATS)
    sim.add_experiment(Experiment(f"N = {n_arms}", agents, environ, N_EPOCHS, n_repeats, plot_position=n_arms))

sim.run_all(save = True)

names = ["Random", "Sparring w/ Thompson Sampling", "Sparring w/ UCB", "DTS", "Thompson Sampling", "UCB", "EXP3", "Epsilon-greedy"]
for i in range(len(N_ARM_VALUES)):
    name = list(sim.experiments.values())[i].name
    sim.experiments[name].name = str(N_ARM_VALUES[i])
sim.plot_aggregated_metrics('copeland_regret', [-10, 10], xlabel="Número de brazos", ylabel = "Regret", title = "Brazos Gaussianos, 10000 épocas", labelsize=18, titlesize=19, legendsize=12, store = True, names = names, storesize = (12,6))
#sim.plot_aggregated_metrics('weak_regret', [-10, 10])
#sim.plot_aggregated_metrics('strong_regret', [-10, 10])
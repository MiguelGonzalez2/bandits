"""
Comparison of sparrings.
"""

from simulation.Simulation import Simulation
from simulation.Experiment import Experiment
from agents.UCBAgent import UCBAgent
from agents.ThompsonBetaAgent import ThompsonBetaAgent
from agents.SparringAgent import SparringAgent
from environments import GaussianEnvironment

N_EPOCHS = 10000
N_REPEATS = 10 # Number of repeats for the largest amount of arms. The remaining arms are run more times since its cheaper.
MAX_N_REPEATS = 200 # Maximum number of repeats

N_ARM_VALUES = [10, 20, 30, 50, 100, 150, 200]

sim = Simulation(f"change arm number, {N_EPOCHS} epochs, different sparrings")

for n_arms in N_ARM_VALUES:
    agents = []
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms), ThompsonBetaAgent(n_arms)))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms, beta_zero = 10), ThompsonBetaAgent(n_arms, beta_zero = 10)))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms, beta_zero = 100), ThompsonBetaAgent(n_arms, beta_zero = 100)))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms, beta_zero = 1000), ThompsonBetaAgent(n_arms, beta_zero = 1000)))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms, beta_zero = 0.1), ThompsonBetaAgent(n_arms, beta_zero = 0.1)))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms, alpha_zero = 10), ThompsonBetaAgent(n_arms, alpha_zero = 10)))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms, alpha_zero = 100), ThompsonBetaAgent(n_arms, alpha_zero = 100)))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms, alpha_zero = 1000), ThompsonBetaAgent(n_arms, alpha_zero = 1000)))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms, alpha_zero = 0.1), ThompsonBetaAgent(n_arms, alpha_zero = 0.1)))
    agents.append(SparringAgent(n_arms, UCBAgent(n_arms),UCBAgent(n_arms)))
    agents.append(SparringAgent(n_arms, UCBAgent(n_arms, exploration_rate=10),UCBAgent(n_arms, exploration_rate=10)))
    agents.append(SparringAgent(n_arms, UCBAgent(n_arms, exploration_rate=0.1),UCBAgent(n_arms, exploration_rate=0.1)))
    agents.append(SparringAgent(n_arms, UCBAgent(n_arms), ThompsonBetaAgent(n_arms)))

    environ = GaussianEnvironment.GaussianEnvironment(n_arms, values = list(range(n_arms)))

    n_repeats = min(int(N_REPEATS * float(N_ARM_VALUES[-1])**2 / float(n_arms)**2), MAX_N_REPEATS)
    sim.add_experiment(Experiment(f"N = {n_arms}", agents, environ, N_EPOCHS, n_repeats, plot_position=n_arms))

sim.run_all(save = True)

sim.plot_aggregated_metrics('copeland_regret', [-10, 10])
sim.plot_aggregated_metrics('weak_regret', [-10, 10])
sim.plot_aggregated_metrics('strong_regret', [-10, 10])
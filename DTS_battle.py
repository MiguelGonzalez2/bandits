"""
Comparison of sparrings.
"""

from simulation.Simulation import Simulation
from simulation.Experiment import Experiment
from agents.DTSAgent import DTSAgent
from environments import GaussianEnvironment

N_EPOCHS = 5000
N_REPEATS = 10 # Number of repeats for the largest amount of arms. The remaining arms are run more times since its cheaper.
MAX_N_REPEATS = 200 # Maximum number of repeats

N_ARM_VALUES = [10, 20, 30, 50, 100, 150, 200]

sim = Simulation(f"change arm number, {N_EPOCHS} epochs, different DTS")

for n_arms in N_ARM_VALUES:
    agents = []

    agents.append(DTSAgent(n_arms, alpha=0.1))
    agents.append(DTSAgent(n_arms))
    agents.append(DTSAgent(n_arms, alpha=10))
    agents.append(DTSAgent(n_arms, alpha=100))
    agents.append(DTSAgent(n_arms, beta=0.1))
    agents.append(DTSAgent(n_arms, beta=10))
    agents.append(DTSAgent(n_arms, beta=100))
    agents.append(DTSAgent(n_arms, gamma=0.1))
    agents.append(DTSAgent(n_arms, gamma=10))
    agents.append(DTSAgent(n_arms, gamma=100))

    environ = GaussianEnvironment.GaussianEnvironment(n_arms, values = list(range(n_arms)))

    n_repeats = min(int(N_REPEATS * float(N_ARM_VALUES[-1])**2 / float(n_arms)**2), MAX_N_REPEATS)
    sim.add_experiment(Experiment(f"N = {n_arms}", agents, environ, N_EPOCHS, n_repeats, plot_position=n_arms))

sim.run_all(save = True)

sim.plot_aggregated_metrics('copeland_regret', [-10, 10])
sim.plot_aggregated_metrics('weak_regret', [-10, 10])
sim.plot_aggregated_metrics('strong_regret', [-10, 10])
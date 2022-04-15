"""
Comparison of sparrings.
"""

from simulation.Simulation import Simulation
from simulation.Experiment import Experiment
from agents.ThompsonBetaAgent import ThompsonBetaAgent
from agents.SparringAgent import SparringAgent
from environments import GaussianEnvironment

N_EPOCHS = 10000
N_REPEATS = 10 # Number of repeats for the largest amount of arms. The remaining arms are run more times since its cheaper.
MAX_N_REPEATS = 200 # Maximum number of repeats

N_ARM_VALUES = [100]

sim = Simulation(f"Thompson Sampling Gridsearch, {N_EPOCHS} epochs, {N_ARM_VALUES[0]} arms.")

alphas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000, 5000, 10000]
betas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000, 5000, 10000]
agents = []

for n_arms in N_ARM_VALUES:

    for beta in alphas:
        for alpha in betas:
            agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms, alpha_zero=alpha, beta_zero=beta), ThompsonBetaAgent(n_arms, alpha_zero=alpha, beta_zero=beta)))

    environ = GaussianEnvironment.GaussianEnvironment(n_arms, values = list(range(n_arms)))

    n_repeats = min(int(N_REPEATS * float(N_ARM_VALUES[-1])**2 / float(n_arms)**2), MAX_N_REPEATS)
    sim.add_experiment(Experiment(f"N = {n_arms}", agents, environ, N_EPOCHS, n_repeats, plot_position=n_arms))

sim.run_all(save = True)


list(sim.experiments.values())[0].plot_metric_grid('copeland_regret', rows = len(alphas), columns = len(betas), xlabels=[str(x) for x in alphas], ylabels=[str(x) for x in betas], xlabel = "Alpha", ylabel= "Beta")
#sim.plot_aggregated_metrics('copeland_regret', [-10, 10])
#sim.plot_aggregated_metrics('weak_regret', [-10, 10])
#sim.plot_aggregated_metrics('strong_regret', [-10, 10])
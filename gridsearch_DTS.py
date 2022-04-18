"""
Comparison of sparrings.
"""

from simulation.Simulation import Simulation
from simulation.Experiment import Experiment
from agents.DTSAgent import DTSAgent
from environments import NoisyGaussianEnvironment

N_EPOCHS = 10000
N_REPEATS = 10 # Number of repeats for the largest amount of arms. The remaining arms are run more times since its cheaper.
MAX_N_REPEATS = 200 # Maximum number of repeats

N_ARM_VALUES = [10]

sim = Simulation(f"DTS Gridsearch, d=2, {N_EPOCHS} epochs, {N_ARM_VALUES[0]} arms.")

def str2(int):
    result = str(int)
    if result[-3:] == "000":
        result = result[:-3] + "k"
    return result

alphas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000, 5000, 10000]
betas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 1000, 5000, 10000]
gammas = [0.1, 1, 10, 100]

for gamma in gammas:
    for n_arms in N_ARM_VALUES:
        agents = []
        for beta in alphas:
            for alpha in betas:
                agents.append(DTSAgent(n_arms=n_arms, alpha = alpha, beta = beta, gamma = gamma))

        environ = NoisyGaussianEnvironment.NoisyGaussianEnvironment(n_arms, d = 2)
        n_repeats = min(int(N_REPEATS * float(N_ARM_VALUES[-1])**2 / float(n_arms)**2), MAX_N_REPEATS)
        sim.add_experiment(Experiment(f"gamma = {gamma}", agents, environ, N_EPOCHS, n_repeats, plot_position=n_arms))

sim.run_all(save = True)

for i in range(len(gammas)):
    list(sim.experiments.values())[i].plot_metric_grid('copeland_regret', title=f"DTS, {N_ARM_VALUES[0]} brazos, 10000 épocas, gamma = {gammas[i]}",rows = len(alphas), columns = len(betas), xlabels=[str2(x) for x in alphas], ylabels=[str2(x) for x in betas], xlabel = "Alpha", ylabel= "Beta", labelsize=10, titlesize=10, store = True, storesize = (6,6))
#sim.plot_aggregated_metrics('copeland_regret', [-10, 10])
#sim.plot_aggregated_metrics('weak_regret', [-10, 10])
#sim.plot_aggregated_metrics('strong_regret', [-10, 10])
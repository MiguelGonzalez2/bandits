"""
Simulation for arm number impact in DB.
"""

from matplotlib.pyplot import xlabel
from simulation.Simulation import Simulation
from simulation.Experiment import Experiment
from agents.MultiSBMAgent import MultiSBMAgent
from agents.ThompsonBetaAgent import ThompsonBetaAgent
from agents.IFAgent import IFAgent
from agents.BTMAgent import BTMAgent
from agents.DoublerAgent import DoublerAgent
from agents.SparringAgent import SparringAgent
from agents.DTSAgent import DTSAgent
from agents.RUCBAgent import RUCBAgent
from agents.CCBAgent import CCBAgent
from agents.RandomAgent import RandomAgent
from environments import GaussianEnvironment

N_EPOCHS = 10000
N_REPEATS = 10 # Number of repeats for the largest amount of arms. The remaining arms are run more times since its cheaper.
MAX_N_REPEATS = 200 # Maximum number of repeats

N_ARM_VALUES = [10, 20, 30, 50, 100, 150, 200]

sim = Simulation(f"change arm number, {N_EPOCHS} epochs, equally spaced arms")


for n_arms in N_ARM_VALUES:
    agents = []
    agents.append(RandomAgent(n_arms))
    agents.append(IFAgent(n_arms, N_EPOCHS))
    agents.append(BTMAgent(n_arms, N_EPOCHS))
    agents.append(DoublerAgent(n_arms, ThompsonBetaAgent(n_arms)))
    agents.append(MultiSBMAgent(n_arms, ThompsonBetaAgent, [n_arms]))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms), ThompsonBetaAgent(n_arms)))
    agents.append(DTSAgent(n_arms))
    agents.append(RUCBAgent(n_arms))
    agents.append(CCBAgent(n_arms))

    environ = GaussianEnvironment.GaussianEnvironment(n_arms, values = list(range(n_arms)))

    n_repeats = min(int(N_REPEATS * float(N_ARM_VALUES[-1])**2 / float(n_arms)**2), MAX_N_REPEATS)
    sim.add_experiment(Experiment(f"N = {n_arms}", agents, environ, N_EPOCHS, n_repeats, plot_position=n_arms))

sim.run_all(save = True)

#list(sim.experiments.values())[1].save_metrics('copeland_regret', xlabel="Época", ylabel = "Regret", title = "20 brazos Gaussianos, 10000 épocas", labelsize=17, titlesize=20, legendsize=12)
names = ["Random", "IF", "BTM", "Doubler", "MultiSBM", "Sparring", "DTS", "RUCB", "CCB"]
for i in range(len(N_ARM_VALUES)):
    name = list(sim.experiments.values())[i].name
    sim.experiments[name].name = str(N_ARM_VALUES[i])
sim.plot_aggregated_metrics('weak_regret', [-10, 10], xlabel="Número de brazos", ylabel = "Regret mínimo", title = "Brazos Gaussianos, 10000 épocas", labelsize=20, titlesize=22, legendsize=12, store = True, names = names, storesize = (12,6))
#sim.plot_aggregated_metrics('weak_regret', [-10, 10])
#sim.plot_aggregated_metrics('strong_regret', [-10, 10])
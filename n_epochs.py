"""
Simulation for epoch impact in DB.
"""

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
from environments import GaussianEnvironment

n_arms = 30
N_REPEATS = 10 # Number of repeats for the largest amount of epochs. The remaining arms are run more times since its cheaper.
MAX_N_REPEATS = 200 # Maximum number of repeats

N_EPOCHS_VALUES = [100, 1000, 2500, 5000, 10000, 25000, 50000, 100000, 150000, 200000]

sim = Simulation(f"change epoch horizon, {n_arms} arms, equally spaced arms")

for n_epochs in N_EPOCHS_VALUES:
    agents = []
    agents.append(IFAgent(n_arms, n_epochs))
    agents.append(BTMAgent(n_arms, n_epochs))
    agents.append(DoublerAgent(n_arms, ThompsonBetaAgent(n_arms)))
    agents.append(MultiSBMAgent(n_arms, ThompsonBetaAgent, [n_arms]))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms), ThompsonBetaAgent(n_arms)))
    agents.append(DTSAgent(n_arms))
    agents.append(RUCBAgent(n_arms))
    agents.append(CCBAgent(n_arms))

    environ = GaussianEnvironment.GaussianEnvironment(n_arms, values = list(range(n_arms)))

    n_repeats = min(int(N_REPEATS * float(N_EPOCHS_VALUES[-1]) / float(n_epochs)), MAX_N_REPEATS)
    sim.add_experiment(Experiment(f"N = {n_epochs}", agents, environ, n_epochs, n_repeats, plot_position=n_epochs))

sim.run_all(save = True)

sim.plot_aggregated_metrics('copeland_regret', padding=[10, 100], cut_ticks = [6000,200001])
sim.save_all_metrics('copeland_regret')
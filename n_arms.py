"""
Simulation for arm number impact in DB.
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

N_EPOCHS = 10000
N_REPEATS = 5 # Number of repeats for the largest amount of arms. The remaining arms are run more times since its cheaper.
MAX_N_REPEATS = 200 # Maximum number of repeats

ARM_VALUES = [10, 20, 30, 50, 100, 150, 200]

sim = Simulation(f"arm number, {N_EPOCHS} epochs")

for n_arms in ARM_VALUES:
    agents = []
    agents.append(IFAgent(n_arms, N_EPOCHS))
    agents.append(BTMAgent(n_arms, N_EPOCHS))
    agents.append(DoublerAgent(n_arms, ThompsonBetaAgent(n_arms)))
    agents.append(MultiSBMAgent(n_arms, ThompsonBetaAgent, [n_arms]))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms), ThompsonBetaAgent(n_arms)))
    agents.append(DTSAgent(n_arms))
    agents.append(RUCBAgent(n_arms))
    agents.append(CCBAgent(n_arms))

    environ = GaussianEnvironment.GaussianEnvironment(n_arms, values = list(range(n_arms)))

    n_repeats = min(int(N_REPEATS * float(ARM_VALUES[-1])**2 / float(n_arms)**2), MAX_N_REPEATS)
    sim.add_experiment(Experiment(f"N = {n_arms}", agents, environ, N_EPOCHS, n_repeats, plot_position=n_arms))

sim.run_all(save = True)

sim.plot_aggregated_metrics('copeland_regret')
sim.save_all_metrics('copeland_regret')
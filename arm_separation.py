"""
Simulation for arm separation values.
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
import numpy as np

N_EPOCHS = 50000
N_REPEATS = 1
n_arms = 2
SEPARATION = [100, 10, 1, 0.1, 0.01, 0.001]
sim = Simulation(f"change arm separation, {N_EPOCHS} epochs, {n_arms} arms")

for separation in SEPARATION:
    agents = []
    agents.append(IFAgent(n_arms, N_EPOCHS))
    agents.append(BTMAgent(n_arms, N_EPOCHS))
    agents.append(DoublerAgent(n_arms, ThompsonBetaAgent(n_arms)))
    agents.append(MultiSBMAgent(n_arms, ThompsonBetaAgent, [n_arms]))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms), ThompsonBetaAgent(n_arms)))
    agents.append(DTSAgent(n_arms))
    agents.append(RUCBAgent(n_arms))
    agents.append(CCBAgent(n_arms))

    environ = GaussianEnvironment.GaussianEnvironment(n_arms, values = np.linspace(1, (n_arms-1)*separation, num=n_arms))

    n_repeats = N_REPEATS
    sim.add_experiment(Experiment(f"s = {separation}", agents, environ, N_EPOCHS, n_repeats, plot_position=separation))

sim.run_all(save = True)

sim.plot_aggregated_metrics('copeland_regret', padding=[0.0005, 0.0005], scale='log')
sim.save_all_metrics('copeland_regret', scale = 'log')
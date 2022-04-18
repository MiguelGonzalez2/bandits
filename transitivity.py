"""
Simulation for epoch impact in DB.
"""

from typing import OrderedDict
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
from environments import NoisyGaussianEnvironment

n_arms = 10
n_repeats = 200
n_epochs = 4000

TRANSITIVITY_VALUES = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

sim = Simulation(f"change transitivity, {n_arms} arms, {n_epochs} epochs")

for transitivity in TRANSITIVITY_VALUES:
    agents = []
    agents.append(RandomAgent(n_arms))
    agents.append(IFAgent(n_arms, n_epochs))
    agents.append(BTMAgent(n_arms, n_epochs))
    agents.append(DoublerAgent(n_arms, ThompsonBetaAgent(n_arms)))
    agents.append(MultiSBMAgent(n_arms, ThompsonBetaAgent, [n_arms]))
    agents.append(SparringAgent(n_arms, ThompsonBetaAgent(n_arms), ThompsonBetaAgent(n_arms)))
    agents.append(DTSAgent(n_arms))
    agents.append(RUCBAgent(n_arms))
    agents.append(CCBAgent(n_arms))

    environ = NoisyGaussianEnvironment.NoisyGaussianEnvironment(n_arms, d = transitivity)

    sim.add_experiment(Experiment(f"d = {transitivity}", agents, environ, n_epochs, n_repeats, plot_position=transitivity))

sim.run_all(save = True)

names = ["Random", "IF", "BTM", "Doubler", "MultiSBM", "Sparring", "DTS", "RUCB", "CCB"]
for i in range(len(TRANSITIVITY_VALUES)):
    name = list(sim.experiments.values())[i].name
    sim.experiments[name].name = str(TRANSITIVITY_VALUES[i])
sim.plot_aggregated_metrics('copeland_regret', [0.2, 0.2], xlabel="Varianza/nivel del ruido", ylabel = "Regret", title = "10 brazos gaussianos ruidosos, 4000 épocas", labelsize=20, titlesize=22, legendsize=12, store = True, names = names, storesize = (12,6))
#sim.plot_aggregated_metrics('copeland_regret', padding=[0.2, 0.2])
#sim.plot_aggregated_metrics('weak_regret', padding=[0.2, 0.2])
#sim.plot_aggregated_metrics('strong_regret', padding=[0.2, 0.2])
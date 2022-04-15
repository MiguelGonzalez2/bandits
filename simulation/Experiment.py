"""
Class representing a MAB/DB experiment.
"""

from .Metrics import Metrics as mm

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random

class Experiment():
    """
    Class that encapsulates all the data needed for a single experiment.
    Here, a "single experiment" means a fixed set of agents against a
    fixed environment, for a fixed amount of iterations wich may be
    repeated more than one time with distinct seeds for averaging.
    """

    def __init__(self, name, agents, environment, n_epochs, n_repeats=1, plot_position = None):
            """
            Initializes the experiment.

            Args:
                Name: identifier for the experiment. Must be unique.
                Agents: list of agents to simulate
                Environment: Environment object with the arms
                n_epochs: Nº of iterations per agent on a given environment
                n_repeats: Nº of environments per agent for robustness
                plot_position: If this experiment can be parameterized within the simulation by a cardinal value 
                    (for example, the number of arms), it should be indicated here for consistent plots.
            """
            self.name = name
            self.agents = agents
            self.environment = environment
            self.n_epochs = n_epochs
            self.metrics = [mm(n_epochs) for i in range(len(agents))]
            self.n_repeats = n_repeats
            self.ran = False
            self.plot_position = plot_position

    def run(self):
        """
        Runs the experiment and stores the metrics.
        """
        # Loops through the several environments
        for _ in tqdm(range(self.n_repeats)):

            optimal_arm = self.environment.get_optimal()
            optimal_value = self.environment.get_optimal_value()
            self.environment.reset()

            # Loops through the several agents
            for agent_id, agent in enumerate(self.agents):

                agent.reset()

                # Carry one experiment
                for i in range(self.n_epochs):
                    # MAB's case:
                    if not agent.is_dueling:
                        # Ask the agent for an action
                        arm = agent.step()
                        # Get reward
                        reward = self.environment.step(arm)
                        # Feed agent
                        agent.reward(arm, reward)
                        # Update metrics
                        self.metrics[agent_id].update(i, self.environment, arm, reward, optimal_arm, optimal_value)
                    # DB's case:
                    else:
                        # Ask the agent for an action
                        arm1, arm2 = agent.step()
                        # Get rewards in order to compare
                        reward1, reward2 = self.environment.dueling_step(arm1, arm2)
                        # Feed agent with the result of the comparison only. Ties are broken randomly
                        agent.reward(arm1, arm2, reward1 > reward2 if reward1 != reward2 else random.choice([True, False]))
                        # Update metrics
                        self.metrics[agent_id].update_dueling(i, self.environment, arm1, arm2, reward1, reward2, optimal_arm, optimal_value)
                
                self.environment.soft_reset()
                self.metrics[agent_id].new_iteration()
        
        self.ran = True

    def plot_metrics(self, metric_name, scale="linear"):
        """
        Plots and shows given metric for the experiment.

        Args:
            metric_name: Name of the desired metric within the available ones (check module "Metrics" or readme).
            scale: pyplot scale format for both axes.
        """
        plots = []
        for i in range(len(self.agents)):
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plots += plt.plot(self.metrics[i].get_metrics()[metric_name], label=self.agents[i].get_name())
        plt.legend(plots, [self.agents[i].get_name() for i in range(len(self.agents))])
        plt.xscale(scale)
        plt.title(f"{self.agents[0].n_arms} arms, {self.n_repeats} simulations with {self.n_epochs} epochs each. Environment: {self.environment.get_name()}")
        plt.show()

    def save_metrics(self, metric_name, scale="linear"):
        """
        Plots and stores to png given metric for the experiment.

        Args:
            metric_name: Name of the desired metric within the available ones (check module "Metrics" or readme).
            scale: pyplot scale format for both axes.
        """
        plots = []
        plt.rcParams["figure.figsize"] = (10, 7)
        for i in range(len(self.agents)):
            plt.xlabel('Epoch')
            plt.ylabel(metric_name)
            plots += plt.plot(self.metrics[i].get_metrics()[metric_name], label=self.agents[i].get_name())
        plt.legend(plots, [self.agents[i].get_name() for i in range(len(self.agents))])
        plt.xscale(scale)
        plt.title(f"{self.agents[0].n_arms} arms, {self.n_repeats} simulations with {self.n_epochs} epochs each. Environment: {self.environment.get_name()}")
        plt.savefig(self.name + "_" + metric_name + '.png')
        plt.clf()

    def get_name(self):
        """
        Returns the name of the experiment.

        Returns:
            the name of the experiment.
        """
        return self.name

    def was_run(self):
        return self.ran

    def get_final_values(self, metric_name):
        """
        Returns dictionary "agent_index: value" where the value is the final value for metric_name.

        Args:
            metric_name: Name of the desired metric within the available ones (check module "Metrics" or readme).

        Returns:
            dictionary "agent_index: value" where the value is the final value for metric_name.
        """
        return {i: self.metrics[i].get_metric_result(metric_name) for i in range(len(self.agents))}

    def get_agent_count(self):
        """
        Returns number of agents.

        Returns:
            number of agents.
        """
        return len(self.agents)

    def get_agent_by_index(self, index):
        """
        Returns agent object by given index.

        Args:
            index: index of the desired agent.

        Returns:
            agent object.
        """
        return self.agents[index]

    def get_plot_position(self):
        """
        Returns this object "plot position" value.
        
        Returns:
            this object "plot position" value.
        """
        return self.plot_position

    def plot_metric_grid(self, metric_name, rows = 1, columns = 1, scale='linear', xlabel = None, ylabel = None, title = None, labelsize = 10, titlesize = 10, xlabels = None, ylabels = None):
        """
        Plots grid, intended for gridsearch.

        xlabels -> horizontal labels left to right.
        ylabels -> vertical labels bottom to top.
        """
        # Collect values for each experiment.
        vals = np.array(list(self.get_final_values(metric_name).values()))

        arr = np.zeros((rows, columns))
        for row in range(rows):
            for column in range(columns):
                arr[row, column] = vals[row*columns + column]

        plt.imshow(arr, cmap = plt.cm.Blues, origin="lower")
        plt.xlabel("Experiment", fontsize = labelsize)
        plt.ylabel(metric_name, fontsize = labelsize)
        if xlabel:
            plt.xlabel(xlabel, fontsize = labelsize)
        if ylabel:
            plt.ylabel(ylabel, fontsize = labelsize)
        plt.xscale(scale)
        plt.title(f"Heatmap of experiment {self.name}", fontsize = titlesize)
        if title:
            plt.title(title, fontsize = titlesize)
        if xlabels:
            plt.xticks(range(columns), xlabels)
        if ylabels:
            plt.yticks(range(rows), ylabels)
        plt.colorbar()
        plt.show()
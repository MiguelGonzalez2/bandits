"""
Carries out multi-experiment simulations.
"""

from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pickle

class Simulation():
    """
    Class that carries out MAB and DB experiments.
    """ 

    def __init__(self, name, experiments=[]):
        """
        Initializes class.

        Args:
            Name: name for the global simulation.
            Experiments: list of experiments that will be carried out.
        """
        self.name = name
        self.experiments = OrderedDict()

        # Insert in order
        for e in experiments:
            self.experiments[e.get_name()] = e

    def add_experiment(self, experiment, override = False):
        """
        Adds experiment to the simulation queue. 

        Args:
            experiment: experiment to be added.
            override: If the experiment already exists (i.e, an experiment with
                the same name exists), then it won't be added unless override is
                set to true.
        """
        id = experiment.get_name() 
        if id not in self.experiments or override:
            self.experiments[id] = experiment

    def get_experiment_count(self):
        """
        Returns number of experiments within the simulation.

        Returns:
            number of experiments.
        """
        return len(self.experiments)

    def get_experiment_names(self):
        """
        Returns the names of the experiments within the simulation.

        Returns:
            names of the experiments.
        """
        return self.experiments.keys()

    def get_experiment_by_name(self, name):
        """
        Returns experiment object given its name.

        Args:
            name: name of the experiment.

        Returns:
            experiment matching the given name.
        """
        return self.experiments[name]

    def get_experiment_by_index(self, index):
        """
        Returns experiment object given its index.

        Args:
            index: index of the experiment.

        Returns:
            experiment object.
        """
        return self.experiments.values()[index]

    def run_all(self, save = False):
        """
        Runs every (remaining) experiment.

        Args:
            save: if set to true, simulation state is saved to disk after each experiment.
        """
        counter = 1
        for id, exp in self.experiments.items():
            if exp.was_run():
                print(f"[{counter}/{self.get_experiment_count()}] Experiment {id} was already executed, skipping.")
            else:
                print(f"[{counter}/{self.get_experiment_count()}] Running experiment {id}...")
                exp.run()
                if save:
                    self.save_state()
                    print(f"Saving state...")

            counter += 1

    def save_all_metrics(self, metric_name, scale="linear"):
        """
        For each ran experiment, saves a png with the metric "metric_name" plotted.
        This doesn't plot aggregated metrics, nor saves the results to file other than the image.

        Args:
            metric_name: Name of the desired metric within the available ones (check module "Metrics" or readme).
            scale: pyplot scale format for both axes.
        """
        for id, exp in self.experiments.items():
            if exp.was_run():
                print(f"Storing metric {metric_name} for experiment {id}.")
                exp.save_metrics(metric_name, scale) 
            else:
                print(f"Warning: experiment {id} was not run, so metric {metric_name} cannot be stored.")

    def plot_all_metrics(self, metric_name, scale="linear"):
        """
        For each ran experiment, plots the metric "metric_name".
        This doesn't plot aggregated metrics.

        Args:
            metric_name: Name of the desired metric within the available ones (check module "Metrics" or readme).
            scale: pyplot scale format for both axes.
        """
        for id, exp in self.experiments.items():
            if exp.was_run():
                print(f"Plotting metric {metric_name} for experiment {id}.")
                exp.plot_metrics(metric_name, scale) 
            else:
                print(f"Warning: experiment {id} was not run, so metric {metric_name} cannot be plotted.")

    def save_state(self):
        """
        Stores the state of the simulation as pickle.
        """
        with open(self.name + ".pkl", "wb") as f:
            pickle.dump(self, f)

    def load_state(self):
        """
        Reloads the state of a pickled simulation and returns it.

        Returns:
            loaded simulation object.
        """
        with open(self.name + ".pkl", "rb") as f:
            return pickle.load(f)

    def plot_aggregated_metrics(self, metric_name, padding=None, scale='linear', cut_ticks = None, xlabel = None, ylabel = None, title = None, labelsize = 10, titlesize = 10, names = None):
        """
        Plots the final value of the given metric for each experiment.
        The values are plotted left to right in order of insertion.
        Make sure that "equivalent" agents are inserted in the same order
        on each experiment so that they get connected if required.

        Args:
            metric_name: Name of the desired metric within the available ones (check module "Metrics" or readme).
            padding: range for x axis will be [min-padding[0], max+padding[1]].
            scale: pyplot scale format for both axes.
            cut_ticks: Dont create xticks for x values smaller than cut_ticks[0] or larger than cut_ticks[1].
        """
        # Collect values for each experiment.
        vals = []
        experiment_names = []
        x_values = []
        for id, exp in self.experiments.items():
            if exp.was_run():
                vals.append(exp.get_final_values(metric_name))
                experiment_names.append(id)
                xpos = exp.get_plot_position()
                x_values.append(xpos if xpos is not None else (max(x_values)+1 if x_values else 1))
            else:
                print(f"Warning: experiment {id} was not run, so metric {metric_name} cannot be plotted.")

        num_agents = min([e.get_agent_count() for e in self.experiments.values()])
        plots = []
        name_counter = 0

        # Set color range to avoid repetition
        colormap = plt.cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1, num_agents)]
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
        
        for i in range(num_agents):
            if names and name_counter < len(names):
                plots += plt.plot(x_values, [v[i] for v in vals], "o--", label=names[name_counter])
                name_counter += 1
            else:
                plots += plt.plot(x_values, [v[i] for v in vals], "o--", label=exp.get_agent_by_index(i).get_name())

        plt.xlabel("Experiment", fontsize = labelsize)
        plt.ylabel(metric_name, fontsize = labelsize)
        if xlabel:
            plt.xlabel(xlabel, fontsize = labelsize)  
        if ylabel:
            plt.ylabel(ylabel, fontsize = labelsize) 
        plt.xscale(scale)
        if padding:
            plt.xlim([min(x_values)-padding[0], max(x_values)+padding[1]])
        plt.xticks(x_values if not cut_ticks else [x for x in x_values if x >= cut_ticks[0] and x <= cut_ticks[1]], 
                    labels=experiment_names if not cut_ticks else [experiment_names[i] for i in range(len(experiment_names)) 
                    if x_values[i] >= cut_ticks[0] and x_values[i] <= cut_ticks[1]])
        plt.legend()
        plt.title(f"Results of simulation \'{self.name}\' with {self.get_experiment_count()} experiments", fontsize = titlesize)
        if title:
            plt.title(title, fontsize = titlesize)
        plt.show()

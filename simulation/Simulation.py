"""
Simulation Environment
"""

from collections import OrderedDict
import matplotlib.pyplot as plt

import pickle

class Simulation():
    """Class that carries out MAB and DB experiments.""" 

    def __init__(self, name, experiments=[]):
        """
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
        experiment: experiment to be added.
        override: If the experiment already exists (i.e, an experiment with
        the same name exists), then it won't be added unless override is
        set to true.
        """
        id = experiment.get_name() 
        if id not in self.experiments or override:
            self.experiments[id] = experiment

    def get_experiment_count(self):
        return len(self.experiments)

    def get_experiment_names(self):
        return self.experiments.keys()

    def get_experiment_by_name(self, name):
        return self.experiments[name]

    def get_experiment_by_index(self, index):
        return self.experiments.values()[index]

    def run_all(self):
        """
        Runs every remaining experiment.
        """
        counter = 1
        for id, exp in self.experiments.items():
            if exp.was_run():
                print(f"[{counter}/{self.get_experiment_count()}] Experiment {id} was already executed, skipping.")
            else:
                print(f"[{counter}/{self.get_experiment_count()}] Running experiment {id}...")
                exp.run()
            counter += 1

    def save_all_metrics(self, metric_name, scale="linear"):
        """
        For each ran experiment, saves a png with the metric "metric_name" plotted.
        This doesn't plot aggregated metrics, nor saves the results to file other than the image.
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
        """
        with open(self.name + ".pkl", "rb") as f:
            return pickle.load(f)

    def plot_aggregated_metrics(self, metric_name):
        """
        Plots the final value of the given metric for each experiment.
        The values are plotted left to right in order of insertion.
        Make sure that "equivalent" agents are inserted in the same order
        on each experiment so that they get connected if required.
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
                x_values.append(xpos if xpos else (max(x_values)+1 if x_values else 1))
            else:
                print(f"Warning: experiment {id} was not run, so metric {metric_name} cannot be plotted.")

        num_agents = min([e.get_agent_count() for e in self.experiments.values()])
        plots = []
        
        for i in range(num_agents):
            plt.xlabel("Experiment")
            plt.ylabel(metric_name)
            plots += plt.plot(x_values, [v[i] for v in vals], "o--", label=exp.get_agent_by_index(i).get_name())
        
        plt.xlim([0.8, len(x_values)+0.2])
        plt.xticks(x_values, labels=experiment_names)
        plt.legend()
        plt.title(f"Results of simulation \'{self.name}\' with {self.get_experiment_count()} experiments")
        plt.show()


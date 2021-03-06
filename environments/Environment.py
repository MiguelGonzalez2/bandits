"""
Generic Environment class.
Can be overriden to support custom implementations.
"""

import numpy as np

class Environment():
    """Implements abstract Environment w/ constant output"""

    def __init__(self, n_arms, value_generator = np.random.normal, values = None):
        """
        Initializes the environment.

        Args:
            n_arms: Number of arms
            value_generator: Function for each arm hidden value (true reward)
            values: Arm values. If given, value_generator is ignored.
        """
        self.value_generator = value_generator
        if values is None:
            self.arms = np.array([value_generator() for i in range(n_arms)])
        else:
            if len(values) > n_arms:
                values = values[:n_arms]
            elif len(values) < n_arms:
                values += [value_generator() for i in range(n_arms - len(values))]
            self.arms = np.array(values, dtype=float)

        self.n_arms = n_arms
        self.pulls = np.zeros(n_arms) # Individual pull values
        self.steps = 0 # Total Steps
        self.probabilities_dueling = np.full((n_arms, n_arms), -1.0) # Cache for pairwise probabilities
        self.copeland_regrets = np.full(n_arms, np.NINF) # Cache for copeland regrets

        # Copeland score of each arm. This measures how many other arms
        # it beats, normalized so that the Condorcet Winner (if any) has score 1.
        self.copeland_scores = np.zeros(n_arms)
        aux = np.array(self.arms, copy=True)
        for i in range(n_arms):
            self.copeland_scores[np.argmax(aux)] = (n_arms-i-1) / (n_arms-1)
            aux[np.argmax(aux)] = np.NINF

    def pull(self, n_arm):
        """
        Pulls a given arm and returns reward. Override in subclasses.

        Args:
            n_arm: arm to be pulled.

        Returns:
            value of the reward obtained.
        """
        return self.arms[n_arm]

    def step(self, n_arm):
        """
        Returns reward for given arm, updating internal values
        This is used for Multi Armed Bandits steps.

        Args:
            n_arm: arm to be pulled.

        Returns:
            value of the reward obtained.
        """
        self.pulls[n_arm] += 1
        self.steps += 1
        return self.pull(n_arm)

    def dueling_step(self, n_arm1, n_arm2):
        """
        Returns rewards for a pair of arms, updating internal values.
        This is used for Dueling Bandits steps. Note that the actual
        rewards are returned so that numerical metrics can be computed
        in order to compare with multi-armed bandits. HOWEVER, dueling
        bandits should never see these values, only the result of the
        pairwise comparison.

        Args:
            n_arm1: first arm of the pair.
            n_arm2: second arm of the pair.

        Returns:
            Tuple containing the reward of each arm.
        """
        self.pulls[n_arm1] += 1
        self.pulls[n_arm2] += 1
        self.steps += 1
        return (self.pull(n_arm1), self.pull(n_arm2))

    def soft_reset(self):
        """
        Only resets metrics but environment is kept the same.
        """
        self.pulls = np.zeros(self.n_arms)
        self.steps = 0
        
    def reset(self):
        """
        Resets environment internals.
        """
        self.soft_reset()
        self.arms = np.array([self.value_generator() for i in range(self.n_arms)])
        self.copeland_scores = np.zeros(self.n_arms)
        self.probabilities_dueling = np.full((self.n_arms, self.n_arms),-1.0)
        self.copeland_regrets = np.full(self.n_arms, np.NINF)
        aux = np.array(self.arms, copy=True)
        for i in range(self.n_arms):
            self.copeland_scores[np.argmax(aux)] = (self.n_arms-i-1) / (self.n_arms-1)
            aux[np.argmax(aux)] = np.NINF

    def get_optimal(self):
        """
        Returns the index of the best arm in the environment.

        Returns:
            index of the best arm in the environment.
        """
        return np.argmax(self.copeland_scores)

    def get_optimal_value(self):
        """
        Returns the best value of the environment.

        Returns:
            the best value of the environment.
        """
        return np.max(self.arms)

    def get_copeland_winners(self):
        """
        Returns a numpy array with the arms with highest copeland score.

        Returns:
            a numpy array with the arms with highest copeland score.
        """
        return np.flatnonzero(self.copeland_scores == self.copeland_scores.max())

    def get_copeland_regret(self, arm):
        """
        Returns the copeland individual arm regret.

        Args:
            arm: index of the arm whose regret is to be obtained.

        Returns:
            the copeland regret for the arm.
        """
        if self.copeland_regrets[arm] > np.NINF:
            return self.copeland_regrets[arm]

        winners = self.get_copeland_winners()

        if arm in winners:
            # No regret if it's one of the winners
            self.copeland_regrets[arm] = 0
            return 0

        max = np.NINF
        for winner in winners:
            delta = self.get_probability_dueling_cached(winner, arm) - 1/2
            if delta > max:
                max = delta
        self.copeland_regrets[arm] = max
        return max

    def get_probability_dueling(self, arm1, arm2):
        """
        Receives two arms and returns the probability that arm1 >= arm2.
        This should be overriden depending on the "pull" function, to match
        the distribution. 

        Args:
            arm1: first arm to be compared
            arm2: second arm to be compared

        Returns:
            Probability that arm1 >= arm2.
        """
        return arm1 / (arm1 + arm2)

    def get_probability_dueling_cached(self, arm1, arm2):
        """
        Receives two arms and returns the probability that arm1 >= arm2.
        Uses cached data to save time. 

        Args:
            arm1: first arm to be compared
            arm2: second arm to be compared

        Returns:
            Probability that arm1 >= arm2.
        """
        if self.probabilities_dueling[arm1, arm2] < 0:
            prob = self.get_probability_dueling(arm1, arm2)
            self.probabilities_dueling[arm1, arm2] = prob
            self.probabilities_dueling[arm2, arm1] = 1 - prob
        return self.probabilities_dueling[arm1, arm2]

    def get_name(self):
        """
        String representation of the environment.

        Returns:
            string representing the environment.
        """
        return f"Default"
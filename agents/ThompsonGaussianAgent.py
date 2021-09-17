"""
Thompson Sampling Agent intended for gaussian environments.
As such, it uses the gaussian distribution as its parameter distribution, and
also a inverted gamma for the variance distribution.
"""

import numpy as np
from scipy.stats import invgamma
from .MABAgent import MABAgent

class ThompsonGaussianAgent(MABAgent):
    
    def __init__(self, n_arms, avg_zero=0, k_zero=1, sigma_zero=1, nu_zero=1, optimism=None):
        """
        Initializes the agent with the following statistical parameters:
        - avg_zero is the starting prediction for the reward average
        - k_zero is the starting certainty for avg_zero
        - sigma_zero is the starting prediction for the degrees of freedom of the variance
        - nu_zero is the scale for the degree of the variance parameter
        More on the statistical background (bayesian conjugate for normal distribution with
        unknown mean and variance) can be found on section 3 of the book:
        http://www.stat.columbia.edu/~gelman/book/BDA3.pdf
        Which has been conveniently summarized for the 1-D case in here:
        https://richrelevance.com/2013/07/31/bayesian-analysis-of-normal-distributions-with-python/
        """
        super(ThompsonGaussianAgent,self).__init__(n_arms, optimism)
        self.avg_zero = avg_zero
        self.k_zero = k_zero
        self.sigma_zero = sigma_zero
        self.nu_zero = nu_zero

        # We need to keep track of the sum of squared differences to the mean. Thanks to
        # a statistical identity, this amounts to keeping track of the sum of the squares
        # of the sample, which can be updated iteratively without needing to store all the data.
        self.square_sum = np.zeros(self.n_arms)

    def step(self):
        """(Override) Returns the arm that should be pulled using Thompson Sampling for Gaussian Environments"""
        estimated_params = np.empty(self.n_arms) # Holds the estimated parameters

        # Estimate the parameters
        for arm in range(self.n_arms):
            # Get number of data points:
            n = self.times_explored[arm]

            # Store the average and sum of squared differences for this arm.
            if n != 0:
                average = self.averages[arm]
                ssd = self.square_sum[arm] - 1/n * (average ** 2)
            else:
                # Guess the values if no data points have been seen yet.
                average = self.avg_zero
                ssd = 0

            # Compute the parameters combining prior and data
            k_n = self.k_zero + n
            avg_n = (self.k_zero/k_n) * self.avg_zero + (n/k_n)*average
            nu_n = self.nu_zero + n

            # Obtain intermediate value used in posterior parameters
            aux = self.nu_zero * self.sigma_zero + ssd + (n*self.k_zero*(self.avg_zero - average)**2)/(k_n)

            # Draw the variance from the variance posterior (inverse gamma)
            variance = (aux/2) * invgamma.rvs(nu_n/2)

            # Draw the mean from the mean posterior (normal)
            estimated_params[arm] = np.random.normal(loc=avg_n, scale=np.sqrt(variance)/k_n)

        # Return the arm which was estimated to be best.
        return np.argmax(estimated_params)

    def reward(self, n_arm, reward):
        """Updates the knowledge given the reward"""
        super().reward(n_arm, reward)

        # Update the average of squares:
        self.square_sum[n_arm] += reward**2

    def reset(self):
        super().reset()
        self.square_avg = np.zeros(self.n_arms)
        

    def get_name(self):
        return f"Thompson Sampling MAB using Normal prior w/params=({self.avg_zero},{self.k_zero},{self.sigma_zero},{self.nu_zero}) opt={self.optimism}"
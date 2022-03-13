"""
Sparring (MAB reduction) dueling bandit agent.
First introduced in http://proceedings.mlr.press/v32/ailon14.pdf.
"""

from .DBAgent import DBAgent

class SparringAgent(DBAgent):
    """
    Implements a dueling bandit agent following the sparring policy.
    """ 

    def __init__(self, n_arms, mab1, mab2):
        """
        Initializes Sparring agent. This agent allows a MAB to be used
        with dueling bandits.

        Args:
            n_arms: number of arms.
            mab1: MAB in charge of arm 1. Must be of type MABAgent.
            mab2: MAB in charge of arm 2. Must be of type MABAgent.
        """
        super(SparringAgent,self).__init__(n_arms)
        
        self.mab1 = mab1
        self.mab2 = mab2

        self.mab1.reset()
        self.mab2.reset()

    def reward(self, n_arm_1, n_arm_2, one_wins):
        """
        Updates the knowledge given the reward. Since it's a Dueling Bandit, the reward
        is a boolean indicating whether the first arm wins or not.
        In MultiSBM it feeds the comparison result to the agent in charge of facing arm 1.

        Args:
            n_arm_1: first arm of the pulled pair.
            n_arm_2: second arm of the pulled pair.
            one_wins: boolean indicating whether the first arm won.
        """

        self.mab1.reward(n_arm_1, int(one_wins))
        self.mab2.reward(n_arm_2, int(not one_wins))

    def step(self):
        """
        (Override) Returns the pair that should be matched using Sparring.

        Returns:
            Pair of indices (i,j) that the policy decided to pull.
        """

        return self.mab1.step(), self.mab2.step()

        
    def reset(self):
        """
        Fully resets the agent
        """
        super().reset()
        self.mab1.reset()
        self.mab2.reset()

    def get_name(self):
        """
        String representation of the agent.

        Returns:
            string representing the agent.
        """
        name1 = self.mab1.get_name()
        name2 = self.mab2.get_name()
        return f"Sparring DB: {name1} VS {name2 if name1 != name2 else 'same'}"

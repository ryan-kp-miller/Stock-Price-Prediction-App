"""
Student Name: Ryan Miller
GT User ID: rmiller327
GT ID: 903461824
"""

import numpy as np
from DTLearner import DTLearner


class RTLearner(DTLearner):
    """
        Random Decision Tree Regressor class
        Superclass: DTLearner
    """
    def __init__(self,leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([np.nan]*4)

    def find_split(self, X, Y):
        """
            helper method for the addEvidence method
            picking random feature for split_col

            inputs:
                X: numpy array containing the features to split based on
                Y: numpy array containing the response

            outputs:
                split_col: integer representing the column of X to split on
                split_val: float representing the value to split split_col on;
                           median of the split column
        """
        split_col = np.random.randint(X.shape[1])
        return split_col, np.median(X[:,split_col])

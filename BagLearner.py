"""
Student Name: Ryan Miller
GT User ID: rmiller327
GT ID: 903461824
"""

import numpy as np

class BagLearner:
    """
        Ensemble method for bagging given regressor
    """
    def __init__(self, learner, kwargs = {}, bags = 20, boost = False, verbose = False):
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def author(self):
        return 'rmiller327'

    def addEvidence(self, Xtrain, Ytrain):
        """
            method for training given learner using bootstrap aggregating
        """
        self.learners = []
        n = Xtrain.shape[0]
        for i in range(self.bags):
            bag_idx = np.random.choice(a=n,size=n,replace=True)
            X_bag = Xtrain[bag_idx]
            Y_bag = Ytrain[bag_idx]
            self.learners.append(self.learner(**self.kwargs))
            self.learners[i].addEvidence(X_bag,Y_bag)

    def query(self, Xtest):
        preds = np.zeros((self.bags, Xtest.shape[0]))
        for i in range(self.bags):
            preds[i] = self.learners[i].query(Xtest)
        return np.mean(preds,axis=0)

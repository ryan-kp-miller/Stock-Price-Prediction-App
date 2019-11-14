"""
Student Name: Ryan Miller
GT User ID: rmiller327
GT ID: 903461824
"""

import numpy as np

class DTLearner():
    """
        Decision Tree Regressor class
    """
    def __init__(self,leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([np.nan]*4)

    def author(self):
        return 'rmiller327'

    def find_split(self, X, Y):
        """
            helper method for the addEvidence method
            finding the feature that is most highly correlated with the response

            inputs:
                X: numpy array containing the features to split based on
                Y: numpy array containing the response

            outputs:
                split_col: integer representing the column of X to split on
                split_val: float representing the value to split split_col on;
                           median of the split column
        """
        corrs = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            #checking if column of X is constant (causes div by 0 error)
            if len(np.unique(X[:,i])) > 1:
                corrs[i] = abs(np.corrcoef(Y,X[:,i]))[0,1]
        split_col = np.argmax(corrs)
        return split_col, np.median(X[:,split_col])

    def split_tree(self, X, Y, split_col, split_val):
        """
            helper method for the addEvidence method
            splits X and Y based on the split_col and split_val

            inputs:
                X: numpy array containing the features
                Y: numpy array containing the response
                split_col: integer representing the column of X to split on
                split_val: float representing the value to split split_col on

            outputs:
                X_left: numpy array containing the features with observations where
                        the split_col is below the split_val
                X_left: numpy array containing the features with observations where
                        the split_col is above the split_val
                Y_left: numpy array containing the response with observations
                        corresponding to X_left
                Y_left: numpy array containing the response with observations
                        corresponding to X_right
        """
        split_mask = X[:,split_col] <= split_val
        X_left = X[split_mask,:]
        Y_left = Y[split_mask]
        X_right = X[~split_mask,:]
        Y_right = Y[~split_mask]

        return (X_left, X_right, Y_left, Y_right)

    def build_tree(self, X, Y):
        """
            recursive method for learning splits for Decision Tree Regressor

            inputs:
                Xtrain: numpy array of shape (m,N) containing the features
                Ytrain: numpy array of shape (m,) containing the response

            output: None
        """
        #checking stopping conditions (y only has one unique response value)
        if Y.shape[0] <= self.leaf_size or len(np.unique(Y)) == 1:
            return np.array([np.nan, np.mean(Y), np.nan, np.nan])
        else:
            #selecting split attribute that gives max information gain
            split_col,split_val = self.find_split(X,Y)

            #splitting data based on best split attribute
            (X_left, X_right, Y_left, Y_right) = self.split_tree(X, Y, \
                                                           split_col, split_val)

            #checking if data didn't split (0 obs in left or right datasets)
            if Y_left.shape[0] == 0 or Y_right.shape[0] == 0:
                return np.array([np.nan, np.mean(Y), np.nan, np.nan])

            #recursively training child nodes
            left_tree = self.build_tree(X_left,Y_left)
            right_tree = self.build_tree(X_right,Y_right)
            root = np.array([split_col,split_val,1,int(left_tree.shape[0]/4)+1])

            return np.append(root, np.append(left_tree, right_tree))


    def addEvidence(self, Xtrain, Ytrain):
        """
            wrapper method for learning splits for Decision Tree Regressor

            inputs:
                Xtrain: numpy array of shape (m,N) containing the features
                Ytrain: numpy array of shape (m,) containing the response

            output: None
        """
        self.tree = self.build_tree(Xtrain, Ytrain).reshape((-1,4))

    #classifying one observation using trained tree
    def query_one(self, record):
        depth = 0
        while ~np.isnan(self.tree[depth,0]):
            #deciding split direction (record[split_col] <= split_val)
            if record[int(self.tree[depth,0])] <= self.tree[depth,1]:
                depth+=int(self.tree[depth,2])
            else:
                depth+=int(self.tree[depth,3])
        return self.tree[depth,1]

    #classifying a set of observations
    def query(self,Xtest):
        """
            wrapper method for classifying new data using learned splits

            input:
                Xtest: numpy array containing the features

            output:
                Ypred: numpy array containing the predictions for Xtest
        """
        return np.array([self.query_one(i) for i in Xtest])

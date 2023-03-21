#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import roc_auc_score
import random
random.seed(42)
np.random.seed(42)


def read_data(filename: str) -> pd.DataFrame:
    '''
        read the data from the filename. Load the data it in a dataframe and return it
    '''

    d = pd.read_csv(filename) # use pandas read_csv function to read the csv file
    df = pd.DataFrame(data=d)
    return df

def data_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    '''
        Follow all the preprocessing steps mentioned in Problem 2 of HW2 (Problem 2: Coding: Preprocessing the Data.)
        Return the final features and final label in same order
        You may use the same code you ssubmiited for problem 2 of HW2
    '''
    df.dropna(inplace=True) #dropping all the rows with nan values
    features = df.loc[:, df.columns != 'NewLeague'] #NewLeague is the label, so we drop this column to extract the features
    label = df['NewLeague'] # Extract NewLeague column as the label
    categorical = features.select_dtypes(exclude = ['int64', 'float64']) # Categorical features have datatypes other than int and float
    categorical = pd.get_dummies(categorical, columns=['League', 'Division']) # we create one hot enocded categorical features here
    categorical = categorical.loc[:, categorical.columns != 'Player'] # we also need to drop the Player column
    numerical = features.select_dtypes(include = ['int64', 'float64']) # numerical features have datatypes int and float
    features = pd.concat([categorical, numerical], axis=1) # then concatenate categorical and numerical features
    label.replace({'A': 0, 'N': 1}, inplace=True) #we replace the labels with 0 and 1
    x = features
    y = label
    return [x,y]

def data_split(features: pd.DataFrame, label:pd.Series, test_size:float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
        Split 80% of data as a training set and the remaining 20% of the data as testing set
        return training and testing sets in the following order: X_train, X_test, y_train, y_test
    '''
    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.2) #split the dataset to train and test 
    return [x_train, x_test, y_train, y_test]

def train_ridge_regression( x_train: np.ndarray, y_train:np.ndarray, x_test: np.ndarray, y_test: np.ndarray, max_iter: int =int(1e8))-> dict:
    '''
        Instantiate an object of Ridge Regression, train the model object using training data for the given N-bootstraps
        iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
        values in all iterations in aucs dictionary

        Rest of the provided handles the return part
    '''
    n_bootstraps = int(1e3)
    aucs         = {"ridge":[]}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    for _ in range(n_bootstraps):
        auc = []
        for lambda_ in lambda_vals:
            estimator = Ridge(max_iter = max_iter, alpha = lambda_) # Instantiate an object of ridge rigression in each iteration with the specific lambda
            estimator.fit(x_train, y_train) # train the model object 
            preds = estimator.predict(x_test) # use the fitted/trained object make predictions for x_test
            auc.append(roc_auc_score(y_test, preds)) #calculate the roc value and append that to the result list
        aucs["ridge"].append(auc)


    print ("ridge mean AUCs:")
    ridge_aucs=pd.DataFrame(aucs["ridge"])
    ridge_mean_auc = {}
    ridge_aucs=pd.DataFrame(aucs["ridge"])
    for lambda_val, ridge_auc in zip(lambda_vals, ridge_aucs.mean()):
        ridge_mean_auc[lambda_val] = ridge_auc
        print ("lambda:", lambda_val, "AUC:", "%.4f"%ridge_auc)
    return ridge_mean_auc

def train_lasso(x_train: np.ndarray, y_train:np.ndarray, x_test: np.ndarray, y_test: np.ndarray, max_iter=int(1e8)) -> dict:
    '''
        Instantiate an object of Lasso Model, train the object using training data for the given N-bootstraps
        iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
        values in all iterations in aucs dictionary

        Rest of the provided handles the return part
    '''
    n_bootstraps = int(1e3)
    aucs         = {"lasso":[]}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    for _ in range(n_bootstraps):
        auc = []
        for lambda_ in lambda_vals:
            estimator = Lasso(max_iter = max_iter, alpha = lambda_) # Instantiate an object of Lasso rigression in each iteration with the specific lambda
            estimator.fit(x_train, y_train) # train the model object 
            preds = estimator.predict(x_test) # use the fitted/trained object make predictions for x_test
            auc.append(roc_auc_score(y_test, preds)) #calculate the roc value and append that to the result list
        aucs["lasso"].append(auc)

    print ("lasso mean AUCs:")
    lasso_mean_auc = {}
    lasso_aucs=pd.DataFrame(aucs["lasso"])
    for lambda_val, lasso_auc in zip(lambda_vals, lasso_aucs.mean()):
        lasso_mean_auc[lambda_val] = lasso_auc
        print ("lambda:", lambda_val, "AUC:", "%.4f"%lasso_auc)
    return lasso_mean_auc

def ridge_coefficients(x_train: np.ndarray, y_train:np.ndarray, optimal_alpha:float, max_iter=int(1e8)) -> Tuple[Ridge, np.ndarray]:
    '''
        return the tuple consisting of trained Ridge model with alpha as optimal_alpha and the coefficients
        of the model
    '''

    model_R = Ridge(max_iter = max_iter, alpha = optimal_alpha) # Instantiate an object of ridge rigression in each iteration with the specific lambda
    model_R.fit(x_train,y_train) # train the model object 
    return [model_R, model_R.coef_] # Return the coefficients


def lasso_coefficients(x_train: np.ndarray, y_train:np.ndarray, optimal_alpha:float, max_iter=int(1e8)) -> Tuple[Lasso, np.ndarray]:
    '''
        return the tuple consisting of trained Lasso model with alpha as optimal_alpha and the coefficients
        of the model
    '''

    model_L = Lasso(max_iter = max_iter, alpha = optimal_alpha) # Instantiate an object of Lasso rigression in each iteration with the specific lambda
    model_L.fit(x_train,y_train) # train the model object 
    return[model_L, model_L.coef_] # Return the coefficients

def ridge_area_under_curve(model_R, x_test: np.ndarray, y_test: np.ndarray) -> float:
    '''
        return area under the curve measurements of trained Ridge model used to find coefficients,
        i.e., model tarined with optimal_aplha
        Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    '''

    score_1 = model_R.predict(x_test) # Use the trained/fitted model to make predictions
    ridge_auc = roc_auc_score(y_test, score_1) #Calculate the roc values 
    return ridge_auc

def lasso_area_under_curve(model_L, x_test: np.ndarray, y_test: np.ndarray) -> float:
    '''
        return area under the curve measurements of Lasso Model,
        i.e., model tarined with optimal_aplha
        Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    '''
    score_2 = model_L.predict(x_test) # Use the trained/fitted model to make predictions
    lasso_auc = roc_auc_score(y_test, score_2) #Calculate the roc values 
    return lasso_auc



class Node:
    def __init__(self, split_val, data = None, left = None, right = None): #left right should be node
        self.left = left
        self.right = right
        self.split_val = split_val # value (of a variable) on which to split. For leaf nodes this is label/output value
        self.data = data #data can be anything! we recommend dictionary with all variables you need



class TreeRegressor():
    def __init__(self, data:np.ndarray, max_depth:int) -> None:
        self.data = data # last element of each row in data is the target variable
        self.max_depth = max_depth # maximum depth
        # YOU MAY ADD ANY OTHER VARIABLES THAT YOU NEED HERE
        # YOU MAY ALSO ADD FUNCTIONS **WITHIN CLASS or functions INSIDE CLASS** TO HELP YOU ORGANIZE YOUR BETTER


    # Build the recursive
    def build_tree(self) -> Node:
        ## return root node
        root = self.get_best_split(self.data) # we call the get_best_split function on the all the data to find the root
        self.split(root, 1)
        return root

    # Calculate the mean squared error for a split dataset
    def mean_squared_error(self, left_split, right_split):
        splits = [left_split, right_split]
        mse = 0.0
        for group in splits:
            size = float(len(group)) # saving the size of each group as we need to use that as the denomneator in the mse calculation
            # avoid divide by zero
            if size == 0:
                continue
            mean_group = self.to_terminal(group) # We need to find out the group of nodes that we need to compute the mse based on
            mse += sum([(row[-1] - mean_group)**2 for row in group]) / size # for each group of split nodes we have in the [left_split, right_split], we compute the mse and add the value to the summation
        return mse

    def split(self, node, depth):
        left, right = node.data['groups']
        
        if not left or not right: # check for a no split
            node.left = node.right = self.to_terminal_node(left + right)
            return
        
        if depth >= self.max_depth: # check for max depth
            node.left, node.right = self.to_terminal_node(left), self.to_terminal_node(right)
            return
        
        if len(left) <= 1: # process left child
            node.left = self.to_terminal_node(left)
        else:
            node.left = self.get_best_split(left) # As the size of the left subtree > 1, we need to split the left sub-tree
            self.split(node.left, depth+1)
        
        if len(right) <= 1:
            node.right = self.to_terminal_node(right) # process right child
        else:
            node.right = self.get_best_split(right) # As the size of the right subtree > 1, we need to split the left sub-tree
            self.split(node.right, depth+1)

    # Select the best split point for a dataset, create a node, and return
    def get_best_split(self, data):
        b_index, b_value, b_score, b_groups = 999, 999, 999, None #Initilize the variables 
        for index in range(len(data[0])-1):
            for row in data:
                groups = self.one_step_split(index, row[index], data) # find the target group nodes using the one_step_split function
                mse = self.mean_squared_error(groups[0], groups[1])  # calculate the mse of that target group of nodes
                if mse < b_score: # if the mse is smaller than the best split score, we need to update that
                    b_index, b_value, b_score, b_groups = index, row[index], mse, groups # update all the variables accordingly
        node = Node(b_value, data = {'index':b_index, 'value':b_value, 'groups':b_groups}) # create the node based on the best values we found in the previos step
        return node

    
    def to_terminal(self, group): # Helper function to find the outcome of the terminal group and the mean value of all the nodes in that group 
        outcomes = [row[-1] for row in group]
        return np.mean(outcomes).item()

    def to_terminal_node(self, group): # Helper function to find the outcome of the terminal group and the value of that node
        val = self.to_terminal(group)
        return Node(val, {'value':val})

    # Split a dataset based on an attribute and an attribute value
    # index is the variable to be split on
    def one_step_split(self, index, value, data):
        left, right = list(), list()
        for row in data:
            if row[index] < value: # we need to compare the value of each index we want to split on to the value to decide if we should go left or right
                left.append(row)
            else:
                right.append(row)
        return left, right

def compare_node_with_threshold(node:Node, row:np.ndarray) ->bool:

    if row[node.data['index']] < node.split_val: # Return True if node's value > row's value (of the variable)
        return True
    else: # Else False
        False

def predict(node, row, comparator):

    if node.left == None and node.right == None: # It is important to return the node.split_val in predict. You can use intermediate variable to save this but make sure to update and return split_val correctly here
        return node.split_val

    if comparator(node, row): # based in the comparator result we need to recursively call the predict on the left and right subtree until we reach to a terminal
        return predict(node.left, row, comparator)
    else:
        return predict(node.right, row, comparator)

# TreeClassifier is a derived class of TreeRegressor

class TreeClassifier(TreeRegressor):

    # Calculate the Gini index for a split dataset
    def gini_index(self, left_split, right_split, classes):
        splits = [left_split, right_split]
        n_instances = float(sum([len(group) for group in splits])) # count all samples at split point
        
        # sum weighted Gini index for each group
        gini = 0.0
        for group in splits:
            size = float(len(group))
            
            if size == 0: # avoid divide by zero
                continue
            score = 0.0
            
            for class_val in classes: # score the group based on the score for each class
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances) # weight the group score by its relative size
        return gini

    def get_best_split(self, data):
        class_values = list(set(row[-1] for row in data))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None #Initilize the variables 
        for index in range(len(data[0])-1):
            for row in data:
                groups = self.one_step_split(index, row[index], data)  # find the target group nodes using the one_step_split function
                score = self.gini_index(groups[0], groups[1], class_values) # calculate the gini index of that target group of nodes
                if score < b_score: # if the gini index is smaller than the best split score, we need to update that
                    b_index, b_value, b_score, b_groups = index, row[index], score, groups # update all the variables accordingly
        node = Node(b_value, data = {'index':b_index, 'value':b_value, 'groups':b_groups}) # create the node based on the best values we found in the previos step
        return node

    
    def to_terminal(self, group): # Helper function to calculate the outcome of the terminal group
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

if __name__ == "__main__":

    df = read_data("/Users/chetan/Desktop/Hitters.csv")
    n_bootstraps = int(1e3)
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    max_iter = 1e8
    final_features, final_label = data_preprocess(df)
    x_train, x_test, y_train, y_test = data_split(final_features, final_label, 0.2)
    ridge_mean_acu = train_ridge_regression(x_train, y_train, x_test, y_test)
    lasso_mean_acu = train_lasso(x_train, y_train, x_test, y_test)
    model_R, ridge_coeff = ridge_coefficients(x_train, y_train, 10)
    model_L, lasso_coeff = lasso_coefficients(x_train, y_train, 0.1)
    ridge_auc = ridge_area_under_curve(model_R, x_test, y_test)
    lasso_auc = lasso_area_under_curve(model_L, x_test, y_test)


    #SUB Q1
    csvname = 'noisy_sin_subsample_2.csv' # Place the CSV file in the same directory as this notebook
    data_regress = np.loadtxt(csvname, delimiter = ',')
    data_regress = np.array([[x, y] for x,y in zip(*data_regress)])
    plt.figure()
    plt.scatter(data_regress[:, 0], data_regress[:, 1])
    plt.xlabel("Features, x")
    plt.ylabel("Target values, y")
    plt.show()

    mse_depths = []
    for depth in range(1, 5):
        regressor = TreeRegressor(data_regress, depth)
        tree = regressor.build_tree()
        mse = 0.0
        for data_point in data_regress:
            mse += (data_point[1] - predict(tree, data_point, compare_node_with_threshold))**2
        mse_depths.append(mse/len(data_regress))
    plt.figure()
    plt.plot(mse_depths)
    plt.xlabel("Depth")
    plt.ylabel("MSE")
    plt.show()

    #SUB Q2
    csvname = 'new_circle_data.csv' # Place the CSV file in the same directory as this notebook
    data_class = np.loadtxt(csvname, delimiter = ',')
    data_class = np.array([[x1, x2, y] for x1,x2,y in zip(*data_class)])
    plt.figure()
    plt.scatter(data_class[:, 0], data_class[:, 1], c=-data_class[:, 2], cmap='bwr')
    plt.xlabel("Features, x1")
    plt.ylabel("Features, x2")
    plt.show()

    accuracy_depths = []
    for depth in range(1, 8):
        classifier = TreeClassifier(data_class, depth)
        tree = classifier.build_tree()
        correct = 0.0
        for data_point in data_class:
            correct += float(data_point[2] == predict(tree, data_point,compare_node_with_threshold))
        accuracy_depths.append(correct/len(data_class))
    # Plot the MSE
    plt.figure()
    plt.plot(accuracy_depths)
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.show()



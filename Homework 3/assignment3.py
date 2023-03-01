#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple, List, Optional, Any, Callable, Dict, Union
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import roc_auc_score
import random
from typeguard import typechecked
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

random.seed(42)
np.random.seed(42)


@typechecked
def read_data(filename: str) -> pd.DataFrame:
    """
    Read the data from the filename. Load the data it in a dataframe and return it.
    """
    return pd.read_csv(filename)


@typechecked
def data_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Follow all the preprocessing steps mentioned in Problem 2 of HW2 (Problem 2: Coding: Preprocessing the Data.)
    Return the final features and final label in same order
    You may use the same code you submiited for problem 2 of HW2
    """
    #drop missing values
    df_dropped_player = df.iloc[: , 1:]
    df_dropped = df_dropped_player.dropna()
   
    #seperate label and features
    labels = df_dropped["NewLeague"]
    feature = df_dropped.drop(df_dropped.columns[df_dropped.shape[1] - 1], axis=1)
    
    #separate numerical columns from nonumerical column
    numbers = feature.select_dtypes(include=['int64', 'float64'])
    # select everything else
    not_numbers = feature.select_dtypes(exclude=['int64', 'float64'])
    # get_dummies, concact, and return
    final_features =  pd.concat([numbers, pd.get_dummies(not_numbers)],axis=1, join='inner')
    
    #transform label into numerical format
    labels.replace('A', 0, inplace=True)
    final_label = labels.replace('N', 1)

    return final_features , final_label


@typechecked
def data_split(
    features: pd.DataFrame, label: pd.Series, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split 80% of data as a training set and the remaining 20% of the data as testing set
    return training and testing sets in the following order: X_train, X_test, y_train, y_test
    """
    trainX, testX = train_test_split(features, test_size=0.2)
    trainY, testY = train_test_split(label, test_size=0.2)
    return trainX, testX, trainY, testY


@typechecked
def train_ridge_regression(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter: int = int(1e8),
) -> Dict[float, float]:
    """
    Instantiate an object of Ridge Regression, train the model object using training data for the given `n'
    iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
    values in all iterations in aucs dictionary

    Rest of the provided handles the return part
    """
    n = int(1e3)
    aucs = {"ridge": []}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]

    for i in range(n):
        for alpha_val in lambda_vals:
            model = Ridge(max_iter=max_iter, alpha=alpha_val)
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            aucs["ridge"].append(roc_auc_score(y_test, prediction))


    print("ridge mean AUCs:")
    ridge_aucs = pd.DataFrame(aucs["ridge"])
    ridge_mean_auc = {}
#    ridge_aucs = pd.DataFrame(aucs["ridge"])
    for lambda_val, ridge_auc in zip(lambda_vals, ridge_aucs.mean()):
        ridge_mean_auc[lambda_val] = ridge_auc
        print("lambda:", lambda_val, "AUC:", "%.4f" % ridge_auc)
    return ridge_mean_auc


@typechecked
def train_lasso(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    max_iter=int(1e8),
) -> Dict[float, float]:
    """
    Instantiate an object of Lasso Model, train the object using training data for the given `n'
    iterations and in each iteration train the model for all lambda_vals as alpha and store roc scores of all lambda
    values in all iterations in aucs dictionary

    Rest of the provided handles the return part
    """
    n = int(1e3)
    aucs = {"lasso": []}
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]


    for i in range(n):
        for alpha_val in lambda_vals:
            model = Lasso(max_iter=max_iter, alpha=alpha_val)
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            aucs["lasso"].append(roc_auc_score(y_test, prediction))


    print("lasso mean AUCs:")
    lasso_mean_auc = {}
    lasso_aucs = pd.DataFrame(aucs["lasso"])
    for lambda_val, lasso_auc in zip(lambda_vals, lasso_aucs.mean()):
        lasso_mean_auc[lambda_val] = lasso_auc
        print("lambda:", lambda_val, "AUC:", "%.4f" % lasso_auc)
    print(lasso_mean_auc)
    return lasso_mean_auc


@typechecked
def ridge_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Ridge, np.ndarray]:
    """
    return the tuple consisting of trained Ridge model with alpha as optimal_alpha and the coefficients
    of the model
    """

    #training
    ridge_model = Ridge(alpha=optimal_alpha, max_iter=max_iter)
    ridge_model.fit(x_train, y_train)
    # get coefficients of the trained model
    coefficients = ridge_model.coef_
    return ridge_model, coefficients


@typechecked
def lasso_coefficients(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    optimal_alpha: float,
    max_iter=int(1e8),
) -> Tuple[Lasso, np.ndarray]:
    """
    return the tuple consisting of trained Lasso model with alpha as optimal_alpha and the coefficients
    of the model
    """
    #training
    lasoo_model = Lasso(alpha=optimal_alpha, max_iter=max_iter)
    lasoo_model.fit(x_train, y_train)

    #coefficients
    coefficients = lasoo_model.coef_
    return lasoo_model , coefficients



@typechecked
def ridge_area_under_curve(model_R, x_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    return area under the curve measurements of trained Ridge model used to find coefficients,
    i.e., model tarined with optimal_aplha
    Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    """
    prediction = model_R.predict(x_test)
    return roc_auc_score(y_test, prediction)


@typechecked
def lasso_area_under_curve(
    model_L, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    return area under the curve measurements of Lasso Model,
    i.e., model tarined with optimal_aplha
    Finally plot the ROC Curve using false_positive_rate, true_positive_rate as x and y axes calculated from roc_curve
    """
    prediction = model_L.predict(x_test)
    return roc_auc_score(y_test, prediction)


class Node:
    @typechecked
    def __init__(
        self,
        split_val: float,
        data: Any = None,
        left: Any = None,
        right: Any = None,
    ) -> None:
        if left is not None:
            assert isinstance(left, Node)

        if right is not None:
            assert isinstance(right, Node)

        self.left = left
        self.right = right
        self.split_val = split_val  # value (of a variable) on which to split. For leaf nodes this is label/output value
        self.data = data  # data can be anything! we recommend dictionary with all variables you need


class TreeRegressor:
    @typechecked
    def __init__(self, data: np.ndarray, max_depth: int) -> None:
        self.data = (
            data  # last element of each row in data is the target variable
        )
        self.max_depth = max_depth  # maximum depth
        # YOU MAY ADD ANY OTHER VARIABLES THAT YOU NEED HERE
        self.root = None
        self.current_depth = None
        # YOU MAY ALSO ADD FUNCTIONS **WITHIN CLASS or functions INSIDE CLASS** TO HELP YOU ORGANIZE YOUR BETTER
        ## YOUR CODE HERE

    @typechecked
    def build_tree(self) -> Node:
        """
        Build the tree
        """
        self.root = self.get_best_split(self.data)
        self.split(self.root, 0)
        return self.root
        

    @typechecked
    def mean_squared_error(self, left_split: np.ndarray, right_split: np.ndarray ) -> float:
        """
        Calculate the mean squared error for a split dataset
        left split is a list of rows of a df, rightmost element is label
        return the sum of mse of left split and right split
        """
        print(left_split.shape)
        print(right_split.shape)
        if left_split.ndim == 1  or right_split.ndim == 1:
                left_mse = np.mean((left_split - np.mean(left_split)) ** 2)
                right_mse = np.mean((right_split - np.mean(right_split)) ** 2)
                return left_mse + right_mse
        else:
            left_y = left_split[:, -1]
            right_y = right_split[:, -1]
            mse = (np.sum((left_y - np.mean(left_y)) ** 2) + np.sum((right_y - np.mean(right_y)) ** 2)) / (len(left_y) + len(right_y))
            return mse

    @typechecked
    def split(self, node: Node, depth: int) -> None:
        """
        Do the split operation recursively
        """
        left_data, right_data = self.one_step_split(node.data["feature_index"], node.split_val, node.data["data"])
        if len(left_data) == 0 or len(right_data) == 0:
            node.left = node.right = Node(np.mean(node.data["threshold"][:, -1]), data=node.data)
            return

        if depth == self.max_depth:
            node.left = Node(np.mean(left_data[:, -1]), data={"data": left_data, "feature_index": None})
            node.right = Node(np.mean(right_data[:, -1]), data={"data": right_data, "feature_index": None})
            return

        node.left = Node(None, data={"data": left_data, "feature_index": None})
        node.right = Node(None, data={"data": right_data, "feature_index": None})
        self.split(node.left, depth + 1)
        self.split(node.right, depth + 1)

    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset AND create a Node
        """
        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        # loop over all the features
        for feature_index in range(data.shape[1]-1):
            feature_values = data[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.one_step_split(feature_index, threshold, data)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = data[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_var_red = self.mean_squared_error(left_y, right_y)
                    # update the best split if needed
                    if curr_var_red>max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        best_split["data"] = y
                        max_var_red = curr_var_red
                        
        # return best split
        return Node(threshold, best_split)

    @typechecked
    def one_step_split(self, index: int, value: float, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split a dataset based on an attribute and an attribute value
        index is the variable to be split on (left split < threshold)
        returns the left and right split each as list
        each list has elements as `rows' of the df
        """
        left_split = []
        right_split = []
        print(data)
        for row in data:
            if row[index] < value:
                left_split.append(row)
            else:
                right_split.append(row)
        return np.array(left_split), np.array(right_split)
    
@typechecked
def compare_node_with_threshold(node: Node, row: np.ndarray) -> bool:
    """
    Return True if node's value > row's value (of the variable)
    Else False
    """
 #   print(node.data, "\n\n\n")
 #   print(node.data["index"])
    index = node.data["index"]
    variable_value = row[index]
    return bool(node.split_val > variable_value)


@typechecked
def predict( node: Node, row: np.ndarray, comparator: Callable[[Node, np.ndarray], bool]) -> float:
    """while node.left or node.right:
        print(isinstance(comparator(node,row).astype(bool)), bool)
        if comparator(node, row):
            node = node.left
        else:
            node = node.right

    #return value at the leaf node
    return node.split_val"""
   
    if node.left is None and node.right is None:
        # leaf node
        return node.split_val

    if (bool(comparator(node, row))):
        return predict(node.left, row, comparator)
    else:
        return predict(node.right, row, comparator)


class TreeClassifier(TreeRegressor):
    def build_tree(self):
        ## Note: You can remove this if you want to use build tree from Tree Regressor
        ######################
        ### YOUR CODE HERE ###
        ######################
        pass

    @typechecked
    def gini_index(
        self,
        left_split: np.ndarray,
        right_split: np.ndarray,
        classes: List[float],
    ) -> float:
        """
        Calculate the Gini index for a split dataset
        Similar to MSE but Gini index instead
        """
        ######################
        ### YOUR CODE HERE ###
        ######################
        pass

    @typechecked
    def get_best_split(self, data: np.ndarray) -> Node:
        """
        Select the best split point for a dataset
        """
        classes = list(set(row[-1] for row in data))
        ######################
        ### YOUR CODE HERE ###
        ######################
        pass


if __name__ == "__main__":
    # Question 1
    """     filename = "hitters.csv"  # Provide the path of the dataset
    df = read_data(filename)
    lambda_vals = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
    max_iter = 1e8
    final_features, final_label = data_preprocess(df)
    x_train, x_test, y_train, y_test = data_split(final_features, final_label, 0.2)
    ridge_mean_acu = train_ridge_regression(x_train, y_train, x_test, y_test)
    lasso_mean_acu = train_lasso(x_train, y_train, x_test, y_test)
    model_R, ridge_coeff = ridge_coefficients(x_train, y_train, 10)
    model_L, lasso_coeff = lasso_coefficients(x_train, y_train, 0.1)
    ridge_auc = ridge_area_under_curve(model_R, x_test, y_test)

    # Plot the ROC curve of the Ridge Model. Include axes labels,
    # legend and title in the Plot. Any of the missing
    # items in plot will result in loss of points.
    ########################
    ## Your Solution Here ##
    ########################
    R_prediction = model_R.predict(x_test)
    R_fpr, R_tpr, R_thresholds = metrics.roc_curve(y_test, R_prediction)
    plt.plot(R_fpr, R_tpr, label="Ridge Regression")
    plt.legend()
    plt.show()
    lasso_auc = lasso_area_under_curve(model_L, x_test, y_test)

    # Plot the ROC curve of the Lasso Model.
    # Include axes labels, legend and title in the Plot.
    # Any of the missing items in plot will result in loss of points.
    L_prediction = model_L.predict(x_test)
    L_fpr, L_tpr, L_thresholds = metrics.roc_curve(y_test, L_prediction)
    plt.plot(L_fpr, L_tpr, label="Lasso")
    plt.legend()
    plt.show()
 """
    # SUB Q1
    data_regress = np.loadtxt("noisy_sin_subsample_2.csv", delimiter=",")
    data_regress = np.array([[x, y] for x, y in zip(*data_regress)])
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
            mse += (data_point[1]- predict(tree, data_point, compare_node_with_threshold)) ** 2
        mse_depths.append(mse / len(data_regress))
    plt.figure()
    plt.plot(mse_depths)
    plt.xlabel("Depth")
    plt.ylabel("MSE")
    plt.show()

    # SUB Q2
    csvname = "new_circle_data.csv"  # Place the CSV file in the same directory as this notebook
    data_class = np.loadtxt(csvname, delimiter=",")
    data_class = np.array([[x1, x2, y] for x1, x2, y in zip(*data_class)])
    plt.figure()
    plt.scatter(
        data_class[:, 0], data_class[:, 1], c=-data_class[:, 2], cmap="bwr"
    )
    plt.xlabel("Features, x1")
    plt.ylabel("Features, x2")
    plt.show()

    accuracy_depths = []
    for depth in range(1, 8):
        classifier = TreeClassifier(data_class, depth)
        tree = classifier.build_tree()
        correct = 0.0
        for data_point in data_class:
            correct += float(data_point[2]== predict(tree, data_point, compare_node_with_threshold))
        accuracy_depths.append(correct / len(data_class))
    # Plot the MSE
    plt.figure()
    plt.plot(accuracy_depths)
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.show()

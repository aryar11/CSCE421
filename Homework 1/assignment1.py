#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from typing import Tuple, List

# Return the dataframe given the filename

# Question 8 sub 1


def read_data(filename: str) -> pd.DataFrame:
    ########################
    ## Your Solution Here ##
    ########################
    data_frame = pd.read_csv(filename)
    return data_frame


# Return the shape of the data

# Question 8 sub 2


def get_df_shape(df: pd.DataFrame) -> Tuple[int, int]:
    ########################
    ## Your Solution Here ##
    ########################
    return df.shape


# Extract features "Lag1", "Lag2", and label "Direction"

# Question 8 sub 3


def extract_features_label(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    ########################
    ## Your Solution Here ##
    ########################
    lag1_lag2 = df.loc[:,['Lag1','Lag2']]
   
    return (lag1_lag2, df["Direction"])
    



# Split the data into a train/test split

# Question 8 sub 4


def data_split(
    features: pd.DataFrame, label: pd.Series, test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ########################
    ## Your Solution Here ##
    ########################
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=test_size)
    return X_train, y_train, X_test, y_test

# df = read_data("Smarket.txt")
# # assert on df
# shape = get_df_shape(df)
# # assert on shape
# features, label = extract_features_label(df)
# x_train, y_train, x_test, y_test = data_split(features, label, 0.33)
# print("train split")
# print(x_train, "\n \n" , y_train, "\n \n \n \n" , x_test, "\n \n", y_test)

# Write a function that returns score on test set with KNNs
# (use KNeighborsClassifier class)

# Question 8 sub 5


def knn_test_score(
    n_neighbors: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    ########################
    ## Your Solution Here ##
    ########################
    knn = KNeighborsClassifier(n_neighbors= n_neighbors)
    knn.fit(x_train, y_train)
    y_prediction = knn.predict(x_test)
    return accuracy_score(y_test, y_prediction)


# Apply k-NN to a list of data
# You can use previously used functions (make sure they are correct)

# Question 8 sub 6


def knn_evaluate_with_neighbours(
    n_neighbors_min: int,
    n_neighbors_max: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> List[float]:
    # Note neighbours_min, neighbours_max are inclusive
    ########################
    ## Your Solution Here ##
    ########################
    accuracy_list = []
    for neighbor in range(n_neighbors_min, n_neighbors_max+1):
        knn = KNeighborsClassifier(n_neighbors= neighbor)
        knn.fit(x_train, y_train)
        y_prediction = knn.predict(x_test)
        accuracy_list.append( accuracy_score(y_test, y_prediction))
    return accuracy_list


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    df = read_data("/home/sumedhpendurkar/ML421/Assignment 1/Smarket 2.csv")
    # assert on df
    shape = get_df_shape(df)
    # assert on shape
    features, label = extract_features_label(df)
    x_train, y_train, x_test, y_test = data_split(features, label, 0.33)
    print(knn_test_score(1, x_train, y_train, x_test, y_test))
    acc = knn_evaluate_with_neighbours(1, 10, x_train, y_train, x_test, y_test)
    plt.plot(range(1, 11), acc)
    plt.show()

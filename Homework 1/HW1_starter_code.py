#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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


def get_df_shape(filename: str) -> Tuple[int, int]:
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
    pass


# Split the data into a train/test split

# Question 8 sub 4


def data_split(
    features: pd.DataFrame, label: pd.Series, test_size: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ########################
    ## Your Solution Here ##
    ########################
    pass


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
    pass


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
    pass


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

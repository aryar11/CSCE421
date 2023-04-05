#!/usr/bin/env python
# coding: utf-8

print("Acknowledgment:")
print("https://github.com/pritishuplavikar/Face-Recognition-on-Yale-Face-Dataset")

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import glob
from numpy import linalg as la
import random
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import os

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import sklearn
from typing import Tuple, List
from typeguard import typechecked


@typechecked
def qa1_load(folder_path:str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the dataset (tuple of x, y the label).

    x should be of shape [165, 243 * 320]
    label can be extracted from the subject number in filename. ('subject01' -> '01 as label)
    """
    ######################
    ### YOUR CODE HERE ###
    ######################
    images = [] 
    labels = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if not file_path.endswith(".txt"):
            #get label
            subjectNumber = int(file.split('.')[0].replace('subject', ''))
            labels.append(subjectNumber)
            #get x
            img = mpimg.imread(file_path)         
            #flat the image into a 1D array of 243*320
            img = img.flatten()
            images.append(img)

    X = np.array(images)
    y = np.array(labels)
    return X, y

@typechecked
def qa2_preprocess(dataset:np.ndarray) -> np.ndarray:
    """
    returns data (x) after pre processing

    hint: consider using preprocessing.MinMaxScaler
    """
    scaler = preprocessing.MinMaxScaler()
    dataset_scaled = scaler.fit_transform(dataset)
    return dataset_scaled

@typechecked
def qa3_calc_eig_val_vec(dataset:np.ndarray, k:int)-> Tuple[PCA, np.ndarray, np.ndarray]:
    """
    Calculate eig values and eig vectors.
    Use PCA as imported in the code to create an instance
    return them as tuple PCA, eigen_value, eigen_vector
    """
    pca = PCA(n_components=k)
    pca.fit(dataset)
    eig_values = pca.explained_variance_
    eig_vectors = pca.components_
    return pca, eig_values, eig_vectors

def qb_plot_written(eig_values:np.ndarray):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots
    """
    plt.plot(range(1, len(eig_values) + 1), eig_values, 'o-')
    plt.xlabel('Number of components')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue vs. Number of Components')
    plt.show()

@typechecked
def qc1_reshape_images(pca:PCA, dim_x = 243, dim_y = 320) -> np.ndarray:
    """
    reshape the pca components into the shape of original image so that it can be visualized
    """
    n_components = pca.n_components_
    images = np.zeros((n_components, dim_x, dim_y))
    for i, comp in enumerate(pca.components_):
        images[i] = comp.reshape((dim_x, dim_y))
    return images

def qc2_plot(org_dim_eig_faces:np.ndarray):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots
    """
    fig, ax = plt.subplots(2, 5, figsize=(20, 8))
    ax = ax.flatten()
    for i in range(10):
        ax[i].imshow(org_dim_eig_faces[i])
        ax[i].set_title(f'Eigenface {i+1}')
    plt.tight_layout()
    plt.show()

@typechecked
def qd1_project(dataset:np.ndarray, pca:PCA) -> np.ndarray:
    """
    Return the projection of the dataset 
    NOTE: TO TEST CORRECTNESS, please submit to autograder
    """
    return pca.transform(dataset)

@typechecked
def qd2_reconstruct(projected_input:np.ndarray, pca:PCA) -> np.ndarray:
    """
    Return the reconstructed image given the pca components
    NOTE: TO TEST CORRECTNESS, please submit to autograder
    """
    return pca.inverse_transform(projected_input)

def qd3_visualize(dataset:np.ndarray, pca:PCA, dim_x = 243, dim_y = 320):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots. You can use other functions that you coded up for the assignment
    """
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qe1_svm(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train svm (with 5 fold cross validation) and return
    best_k, and test accuracy (averaged across fold).

    Hint: you can pick 5 `k' values uniformly distributed
    """
    ######################
    ### YOUR CODE HERE ###
    ######################

@typechecked
def qe2_lasso(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train svm (with 5 fold cross validation) and return
    best_k, and test accuracy (averaged across fold) in that order.

    Hint: you can pick 5 `k' values uniformly distributed
    """
    ######################
    ### YOUR CODE HERE ###
    ######################



if __name__ == "__main__":

    faces, y_target = qa1_load("./data/")
    dataset = qa2_preprocess(faces)
    pca, eig_values, eig_vectors = qa3_calc_eig_val_vec(dataset, len(dataset))

    #qb_plot_written(eig_values)

    num = len(dataset)
    org_dim_eig_faces = qc1_reshape_images(pca)
    #qc2_plot(org_dim_eig_faces)

    qd3_visualize(dataset, pca)
    best_k, result = qe1_svm(dataset, y_target, pca)
    print(best_k, result)
    best_k, result = qe2_lasso(dataset, y_target, pca)
    print(best_k, result)

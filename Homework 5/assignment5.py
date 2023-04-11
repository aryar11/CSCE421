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
from sklearn.model_selection import StratifiedKFold, cross_val_score , KFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, LogisticRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from typing import Tuple, List
from typeguard import typechecked


@typechecked
def qa1_load(folder_path:str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the dataset (tuple of x, y the label).

    x should be of shape [165, 243 * 320]
    label can be extracted from the subject number in filename. ('subject01' -> '01 as label)
    """
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
    #get two random images from the dataset
    img_idx1, img_idx2 = np.random.choice(len(dataset), size=2, replace=False)
    #img1 = dataset[img_idx1]
    #img2 = dataset[img_idx2]
    # set number of components to visualize
    num_components = [1,10, 20, 30, 40, 50, 100]

    #plot first
    plt.figure(figsize=(14, 7))
    for i, k in enumerate(num_components):
        #train pcs for num of components
        pca = PCA(n_components=k)
        pca.fit(dataset)
        #work with img1
        img1_reconstructed = qd2_reconstruct(qd1_project(dataset, pca)[img_idx1], pca).reshape(dim_x, dim_y)
        plt.subplot(2, 7, i+1)
        plt.imshow(img1_reconstructed, cmap='gray')
        plt.axis('off')
        plt.title(f'k={k}')

        #plot image 2
        img2_reconstructed = qd2_reconstruct(qd1_project(dataset, pca)[img_idx2], pca).reshape(dim_x, dim_y)
        plt.subplot(2, 7, i+8)
        plt.imshow(img2_reconstructed, cmap='gray')
        plt.axis('off')
        plt.title(f'k={k}')

    plt.tight_layout()
    plt.show()

    
@typechecked
def qe1_svm(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train svm (with 5 fold cross validation) and return
    best_k, and test accuracy (averaged across fold).

    Hint: you can pick 5 `k' values uniformly distributed
    """
    k_values = np.arange(10, 101, 20)
    k_fold = StratifiedKFold(n_splits=5, shuffle=True)
    best_k = 0
    best_accuracy = 0.0
    
    for k in k_values:
        #project the data onto k PCA components
        projected_trainX = pca.transform(trainX)[:, :k]
        # train   SVM with RBF kernel and cross validate
        svc = SVC(kernel='rbf')
        accuracy = np.mean(cross_val_score(svc, projected_trainX, trainY, cv=k_fold))
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    return int(best_k), best_accuracy


@typechecked
def qe2_lasso(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train svm (with 5 fold cross validation) and return
    best_k, and test accuracy (averaged across fold) in that order.

    Hint: you can pick 5 `k' values uniformly distributed
    """
    #init k values
    num_components = list(range(10, 101, 20))
    best_k = None
    best_acc = 0.0
    
    #split data
    X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size=0.2)
    
    for k in num_components:
        #change num of components
        pca.n_components = k
        pca.fit(X_train)
        train_transformed = pca.transform(X_train)
        test_transformed = pca.transform(X_test)

        #train and do 5 fold CV
        lasso = LassoCV(cv=5).fit(train_transformed, y_train)
        acc = lasso.score(test_transformed, y_test)

        if acc > best_acc:
            best_acc = acc
            best_k = k

    return best_k, best_acc


if __name__ == "__main__":

    faces, y_target = qa1_load("./data/")
    dataset = qa2_preprocess(faces)
    pca, eig_values, eig_vectors = qa3_calc_eig_val_vec(dataset, len(dataset))

    qb_plot_written(eig_values)

    num = len(dataset)
    org_dim_eig_faces = qc1_reshape_images(pca)
    qc2_plot(org_dim_eig_faces)

    qd3_visualize(dataset, pca)
    best_k, result = qe1_svm(dataset, y_target, pca)
    print(best_k, result)
    best_k, result = qe2_lasso(dataset, y_target, pca)
    print(best_k, result)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download and read the data.
def read_data(filename: str) -> pd.DataFrame:
    '''
    Read the data from the given file name and return a pandas dataframe
    '''
    return pd.read_csv(filename)

# Prepare your input data and labels
def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    '''
    Separate input data and labels, remove NaN values. Execute this for both 
    dataframes.
    return tuple of numpy arrays(train_data, train_label, test_data, test_label).
    '''
    # Separate the input features and labels for the training and test data
    train_X = df_train.drop("label", axis=1).values
    train_y = df_train["label"].values
    test_X = df_test.drop("label", axis=1).values
    test_y = df_test["label"].values

    # Remove NaN values from the data
    train_X = train_X[~np.isnan(train_X).any(axis=1)]
    train_y = train_y[~np.isnan(train_y)]
    test_X = test_X[~np.isnan(test_X).any(axis=1)]
    test_y = test_y[~np.isnan(test_y)]

    return train_X, train_y, test_X, test_y

# Implement LinearRegression class
class LinearRegression_Local:   
    def __init__(self, learning_rate=0.00001, iterations=30):        
        self.learning_rate = learning_rate
        self.iterations    = iterations
        self.weights       = None
    
    # Function for model training         
    def fit(self, X, Y):
        # weight initialization
        self.weights = np.zeros(X.shape[1])
        
        for i in range(self.iterations):
            self.update_weights(X, Y)
        
    # Helper function to update weights in gradient descent
    def update_weights(self, X, Y):
        # predict on data and calculate gradients
        y_pred = X.dot(self.weights)
        error = y_pred - Y
        gradient = X.T.dot(error) / X.shape[0]
        
        # update weights
        self.weights -= self.learning_rate * gradient
    
    # Hypothetical function  h( x )
    def predict(self, X):
        return X.dot(self.weights)

# Build your model
def build_model(train_X: np.array, train_y: np.array):
    '''
    Instantiate an object of LinearRegression class, train the model object
    using training data and return the model object
    '''
    model = LinearRegression_Local()
    model.fit(train_X, train_y)

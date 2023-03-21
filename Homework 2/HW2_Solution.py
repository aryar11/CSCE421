################
################
## Q1
################
################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score
from typing import Tuple, List
import scipy.stats


# Download and read the data.
def read_data(filename: str) -> pd.DataFrame:
    '''
        read data and return a dataframe
    '''
    df  = pd.read_csv(filename) # use pandas read_csv function to read the dataframe
    return df

# Prepare your input data and labels
def prepare_data(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple:
    '''
        Separate input data and labels, remove NaN values. Execute this for both dataframes.
        return tuple of numpy arrays(train_data, train_label, test_data, test_label).
    '''

    # extract x and y columns from train and test set

    X_train = np.array(df_train['x']) 
    y_train = np.array(df_train['y'])
    X_test  = np.array(df_test['x'])
    y_test  = np.array(df_test['y'])

    # find the indices where y value is nan

    nan_train = np.where(np.isnan(y_train))
    nan_test  = np.where(np.isnan(y_train))

    # drop the corresponding x values based on the indices we saved in the previous step
    X_train = np.delete(X_train, nan_train)
    y_train = np.delete(y_train, nan_train)
    return (X_train, y_train, X_test, y_test)

# Implement LinearRegression class
class LinearRegression_Local:   
    def __init__(self, learning_rate=0.00001, iterations=30):        
        self.learning_rate = learning_rate
        self.iterations    = iterations
    
    # Function for model training         
    def fit(self, X, Y):
        # weight initialization
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)          
        self.b = 0
        
        # data
        self.X = X         
        self.Y = Y
        
        # gradient descent learning to update the weights in each iteration               
        for i in range(self.iterations) :              
            self.update_weights()  

    # Helper function to update weights in gradient descent      
    def update_weights(self):
        
        Y_pred = self.predict(self.X) # make predictions on data and calculate gradients 
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m  # derivative of weights
        db = - 2 * np.sum(self.Y - Y_pred) / self.m # derivative of bias
        
        self.W = self.W - dW * self.learning_rate # update the weights using the learning rate and derivative we calculated in the previous step
        self.b = self.b - db * self.learning_rate # update the bias using the learning rate and derivative we calculated in the previous step

    # Hypothetical function h(x)       
    def predict(self, X):
        return X.dot(self.W) + self.b

# Build your model
def build_model(train_X: np.array, train_y: np.array):
    '''
        Instantiate an object of LinearRegression class, train the model object
        using training data and return the model object
    '''
    X_train = np.expand_dims(train_X, -1)
    X_test = np.expand_dims(X_test, -1)
    model = LinearRegression_Local(0.0001, 30) #Instantiate the model object based on the LinearRegression class we wrote in the previous step
    model.fit(X_train, train_y) #train/fit the model to the training data
    return model


# Make predictions with test set
def pred_func(model, X_test):
    '''
        return numpy array comprising of prediction on test set using the model
    '''
    return model.predict(X_test) # Use the trained model to make prediction on the test data

# Calculate and print the mean square error of your prediction
def MSE(y_test, pred):
    '''
        return the mean square error corresponding to your prediction
    '''
    val = metrics.mean_squared_error(y_test, pred) # calculate the MSE using the predictions and ground truth (y_test)
    return val

################
################
## Q2
################
################

# Download and read the data.
def read_training_data(filename: str) -> tuple:
    '''
        read train data into a dataframe df1, store the top 10 entries of the dataframe in df2
        and return a tuple of the form (df1, df2, shape of df1)   
    '''
    d = pd.read_csv('Hitters.csv') # use the pandas read_csv function to read the data
    return (d, d.head(10), d.shape) 

# Prepare your input data and labels
def data_clean(df_train: pd.DataFrame) -> tuple:
    '''
        check for any missing values in the data and store the missing values in series s, drop the entries corresponding 
        to the missing values and store dataframe in df_train and return a tuple in the form: (s, df_train)
    '''
    s = df_train.isnull().sum() # find how many missing values the training dataset has
    df_train = df_train.dropna() # drop the nan values

    return (s, df_train)

def feature_extract(df_train: pd.DataFrame) -> tuple:
    '''
        New League is the label column.
        Separate the data from labels.
        return a tuple of the form: (features(dtype: pandas.core.frame.DataFrame), label(dtype: pandas.core.series.Series))
    '''
    features = df_train.loc[:, df.columns != 'NewLeague'] # drop the "NewLeague" column as it is the label to find the features
    label = df_train['NewLeague'] # extract the "NewLeague" column as the label

    return (features, label)

def data_preprocess(feature: pd.DataFrame) -> pd.DataFrame:
    '''
        Separate numerical columns from nonnumerical columns. (In pandas, check out .select dtypes(exclude = ['int64', 'float64']) and .select dtypes(
        include = ['int64', 'float64']). Afterwards, use get dummies for transforming to categorical. Then concat both parts (pd.concat()).
        and return the concatenated dataframe.
    '''
    categorical = features.select_dtypes(exclude = ['int64', 'float64']) # Exclude int and float types to find the categorical features
    categorical = categorical.loc[:, categorical.columns != 'Player'] # drop the player column
    categorical = pd.get_dummies(categorical, columns=['League', 'Division']) # Convert League and Division columns to the one-hot encoding version as they're categorical
    numerical   = features.select_dtypes(include = ['int64', 'float64']) # extract the numerical features where data type is either int or float
    features    = pd.concat([categorical, numerical], axis=1) # Concatenate both categorical and numerical features to create the complete feature list

    return features

def label_transform(labels: pd.Series) -> pd.Series:
    '''
        Transform the labels into numerical format and return the labels
    '''
    label.replace({'A': 0, 'N': 1}, inplace=True) # transform the labels to make them ready for model training and testing
    return label

################
################
## Q3
################
################ 
def data_split(features: pd.DataFrame, label:pd.Series, random_state  = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
        Split 80% of data as a training set and the remaining 20% of the data as testing set using the given random state
        return training and testing sets in the following order: X_train, X_test, y_train, y_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42) # Split the data to have 80% for training and 20% for test
    return [X_train, X_test, y_train, y_test]

def train_linear_regression( x_train: np.ndarray, y_train:np.ndarray):
    '''
        Instantiate an object of LinearRegression class, train the model object
        using training data and return the model object
    '''
    linear = LinearRegression().fit(X_train, y_train) # Instantiate a SKLearn Linear Regression model object and train it with the training data
    return linear

def train_logistic_regression( x_train: np.ndarray, y_train:np.ndarray, max_iter=1000000):
    '''
        Instantiate an object of LogisticRegression class, train the model object
        use provided max_iterations for training logistic model
        using training data and return the model object
    '''
    logistic = LogisticRegression(max_iter=1000000).fit(X_train, y_train) # Instantiate a SKLearn Logistic Regression model object and train it with the training data
    return logistic

def models_coefficients(linear_model, logistic_model) -> Tuple[np.ndarray, np.ndarray]:
    '''
        return the tuple consisting the coefficients for each feature for Linear Regression 
        and Logistic Regression Models respectively
    '''
    return [linear_model.coef_, logistic_model.coef_]   # find and return the coefficient of each model using .coef_

def linear_pred_and_area_under_curve(linear_model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.array, np.array, np.array, np.array, float]:
    '''
        return the tuple consisting the predictions and area under the curve measurements of Linear Regression 
        and Logistic Regression Models respectively in the following order 
        [linear_reg_pred, linear_reg_fpr, linear_reg_tpr, linear_threshold, linear_reg_area_under_curve]
        Finally plot the ROC Curve
    '''
    y_predict = linear_model.predict(x_test) # Use the trained model to make prediction on x_test

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict) # save the false positive rate and true positive rate with the corresponding threshold values
    threshold_linear = thresholds
    lin_f1 = f1_score(y_test, y_predict.round()) # Find the f1 score using the ground truth (y_test) and rounded prediction values

    plt.plot(fpr, tpr, label='linear')
    plt.legend()
    plt.show()

    return [y_predict, fpr, tpr, thresholds, roc_auc_score(y_test, y_predict)]

def logistic_pred_and_area_under_curve(logistic_model, x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.array, np.array, np.array, np.array, float]:
    '''
        return the tuple consisting the predictions and area under the curve measurements of Linear Regression 
        and Logistic Regression Models respectively in the following order 
        [log_reg_pred, log_reg_fpr, log_reg_tpr, log_threshold, log_reg_area_under_curve]
        Finally plot the ROC Curve
    '''
    y_predict = logistic_model.predict_proba(x_test) # use predict_proba to make predictions on x_test
    y_pred = []

    for each in y_predict:
        y_pred.append(each[1]) # just use the either first or second element of the prediction as logistic regression create a pair
    y_pred = np.array(y_pred)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred) # save the false positive rate and true positive rate with the corresponding threshold values
    threshold_log = thresholds

    log_f1 = f1_score(y_test, y_pred.round()) # Find the f1 score using the ground truth (y_test) and rounded prediction values
    plt.plot(fpr, tpr, label='logistic')
    plt.legend()
    plt.show()

    return [y_pred, fpr, tpr, thresholds, roc_auc_score(y_test, y_pred)]



def optimal_thresholds(linear_threshold: np.ndarray, linear_reg_fpr: np.ndarray, linear_reg_tpr: np.ndarray, log_threshold: np.ndarray, log_reg_fpr: np.ndarray, log_reg_tpr: np.ndarray) -> Tuple[float, float]:
    '''
        return the tuple consisting the thresholds of Linear Regression and Logistic Regression Models respectively
    '''
    linear_optimal_idx = np.argmax(linear_reg_tpr - linear_reg_fpr) # Find the index where tpr-fpr is the maximum
    linear_optimal_threshold = linear_threshold[linear_optimal_idx]

    log_optimal_idx = np.argmax(log_reg_tpr - log_reg_fpr) # Find the index where tpr-fpr is the maximum
    log_optimal_threshold = log_threshold[log_optimal_idx]

    return [linear_optimal_threshold, log_optimal_threshold]

def stratified_k_fold_cross_validation(num_of_folds: int, shuffle: True, features: pd.DataFrame, label: pd.Series):
    '''
        split the data into 5 groups. Checkout StratifiedKFold in scikit-learn
    '''

    skf=StratifiedKFold(n_splits=num_of_folds, shuffle=True) # Instantiate a skf object based on the number of folds and return it
    return skf

def train_test_folds(skf, num_of_folds: int, features: pd.DataFrame, label: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    '''
        train and test in for loop with different training and test sets obatined from skf. 
        use a PENALTY of 12 for logitic regression model for training
        find features in each fold and store them in features_count array.
        populate auc_log and auc_linear arrays with roc_auc_score of each set trained on logistic regression and linear regression models respectively.
        populate f1_dict['log_reg'] and f1_dict['linear_reg'] arrays with f1_score of trained logistic and linear regression models on each set
        return features_count, auc_log, auc_linear, f1_dict dictionary
    '''

    # Initilize the variables, list and dictionaries we use in this function to save the result

    num_of_folds = num_of_folds
    max_iter = 100000008

    X = features
    y = label
    auc_log = []
    auc_linear = []
    features_count = []

    f1_dict = {'log_reg': [], 'linear_reg': []}

    for train_index, test_index in skf.split(X, y): 
        X_train, X_test = X.iloc[train_index], X.iloc[test_index] # find the corresponding train and test sets based on the index from SKF
        y_train, y_test = y.iloc[train_index].values.ravel(), y.iloc[test_index].values.ravel()
        features.count.append(X_test.columns.size) # Keep track of the number of features here
        
        log_regressor   = LogisticRegression(penalty='l2', max_iter = max_iter) # Instantiate a logistic regression object
        log_regressor.fit(X_train, y_train) # Train the model with the training data
        
        linear_regressor = LinearRegression() # Instantiate a linear regression object
        linear_regressor.fit(X_train, y_train)  # Train the model with the training data
        
        log_pred   =   log_regressor.predict(X_test) # Make prediction with the x_test data
        linear_pred =  linear_regressor.predict(X_test)  # Make prediction with the x_test data
        
        # Saved the aread under the curve of ROC curves:
        auc_log.append(roc_auc_score(y_test, log_pred)) 
        auc_linear.append(roc_auc_score(y_test, linear_pred))
        
        # Save the F1 score for each model
        f1_dict['log_reg'].append(f1_score(y_test, log_pred.round()))
        f1_dict['linear_reg'].append(f1_score(y_test, linear_pred.round()))

    return [features_count, auc_log, auc_linear, f1_dict]

def is_features_count_changed(features_count: np.array) -> bool:
    '''
        compare number of features in each fold (features_count array's each element)
        return true if features count doesn't change in each fold. else return false
    '''
    is_features_count_changed = True
    for i in range(1, len(features_count)):
        if(features_count[i] != features_count[i-1]): # Check if number of features changes in each iteration
            is_features_count_changed = False
    return is_features_count_changed

def mean_confidence_interval(data: np.array, confidence=0.95) -> Tuple[float, float, float]:
    '''
        To calculate mean and confidence interval, in scipy checkout .sem to find standard error of the mean of given data (AUROCs/ f1 scores of each model, linear and logistic trained on all sets). 
        Then compute Percent Point Function available in scipy and mutiply it with standard error calculated earlier to calculate h. 
        The required interval is from mean-h to mean+h
        return the tuple consisting of mean, mean -h, mean+h
    '''
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a) # compute the mean and standard error of the given data
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1) # Calculate h based on the confidence value and standard error 
    return [m, m-h, m+h]

if __name__ == "__main__":
    ################
    ################
    ## Q1
    ################
    ################
    data_path_train   = "/autograder/source/train.csv"
    data_path_test    = "/autograder/source/test.csv"
    df_train = read_data(data_path_train)
    df_test = read_data(data_path_test)

    train_X, train_y, test_X, test_y = prepare_data(df_train, df_test)
    model = build_model(train_X, train_y)

    # Make prediction with test set
    preds = pred_func(model, test_X)

    # Calculate and print the mean square error of your prediction
    mean_square_error = MSE(test_y, preds)

    # plot your prediction and labels, you can save the plot and add in the report

    plt.plot(test_y, label='label')
    plt.plot(preds, label='pred')
    plt.legend()
    plt.show()

    ################
    ################
    ## Q2
    ################
    ################
    data_path_training   = "/autograder/source/Hitters.csv"

    train_df, df2, df_train_shape = read_training_data(data_path_training)
    s, df_train_mod = data_clean(train_df)
    features, label = feature_extract(df_train_mod)
    final_features  = data_preprocess(features)
    final_label     = label_transform(label)

    ################
    ################
    ## Q3
    ################
    ################

    num_of_folds = 5
    max_iter = 100000008
    X = final_features
    y = final_features
    auc_log = []
    auc_linear = []
    features_count = []
    f1_dict = {'log_reg': [], 'linear_reg': []}
    is_features_count_changed = True

    X_train, X_test, y_train, y_test = data_split(final_features, final_label)

    linear_model = train_linear_regression(X_train, y_train)

    logistic_model = train_logistic_regression(X_train, y_train)

    linear_coef, logistic_coef = models_coefficients(linear_model, logistic_model)

    print(linear_coef)
    print(logistic_coef)

    linear_y_pred, linear_reg_fpr, linear_reg_tpr, linear_reg_area_under_curve, linear_threshold = linear_pred_and_area_under_curve(linear_model, X_test, y_test)

    log_y_pred, log_reg_fpr, log_reg_tpr, log_reg_area_under_curve, log_threshold = logistic_pred_and_area_under_curve(logistic_model, X_test, y_test)

    # plot your prediction and labels
    plt.plot(log_reg_fpr, log_reg_fpr, label='logistic')
    plt.plot(linear_reg_fpr, linear_reg_tpr, label='linear')
    plt.legend()
    plt.show()
    
    linear_threshod, linear_threshod = optimal_thresholds(y_test, linear_y_pred, log_y_pred, linear_threshold, log_threshold)

    skf = skf = stratified_k_fold_cross_validation(num_of_folds, True, features=final_features, label=final_label)
    features_count, auc_log, auc_linear, f1_dict = train_test_folds(skf, num_of_folds, final_features, final_label)

    print("Does features change in each fold?")

    # call is_features_count_changed function and return true if features count changes in each fold. else return false
    is_features_count_changed = is_features_count_changed(features_count)
    linear_threshold, log_threshold = optimal_thresholds(linear_threshold, linear_reg_fpr, linear_reg_tpr, log_threshold, log_reg_fpr, log_reg_tpr)

    print(is_features_count_changed)

    auc_linear_mean, auc_linear_open_interval, auc_linear_close_interval = 0, 0, 0
    auc_log_mean, auc_log_open_interval, auc_log_close_interval = 0, 0, 0

    f1_linear_mean, f1_linear_open_interval, f1_linear_close_interval = 0, 0, 0
    f1_log_mean, f1_log_open_interval, f1_log_close_interval = 0, 0, 0


    #Find mean and 95% confidence interval for the AUROCs for each model and populate the above variables accordingly
    #Hint: use mean_confidence_interval function and pass roc_auc_scores of each fold for both models (ex: auc_log)
    #Find mean and 95% confidence interval for the f1 score for each model.

    mean_confidence_interval(auc_log)
    mean_confidence_interval(auc_linear)
    mean_confidence_interval(f1_dict['log_reg'])
    mean_confidence_interval(f1_dict['linear_reg'])












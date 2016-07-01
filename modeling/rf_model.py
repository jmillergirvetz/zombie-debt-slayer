###Module to model - LogRegCV, RF, XGBC, SVM with poly & rbf kernels###
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

import cPickle as pickle


def load_pickle(picklefile):
    """
    Function that loads in pickled feature matrix
    INPUT: pickle file path
    OUTPUT: feature matrix
    """
    with open(picklefile, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, picklefile):
    """
    Function that saves either a pickled object or pickled model
    INPUT: path to file and filename and the fitted model to pickle
    OUTPUT: pickled file to the specified path
    """
    with open(picklefile, 'wb') as f:
        pickle.dump(data, f)

def pred_mat(X_test, model_list):
    """
    Function that creates numpy matrix of predictions for each fitted model
    INPUT: X_test set used for model predictions and a list of fitted models
    OUTPUT: list of numpy arrays containing predictions for each model
    """
    L = []
    for model in model_list:
        L.append(model.predict(X_test))
    return L

def combine_model_pred(model_list, top_model=1):
    """
    Function that creates numpy matrix of predictions for each fitted model
    INPUT: list of numpy arrays containing predictions and the top model column
    OUTPUT: a numpy array containing the aggregated major predictions \
    """
    maj_pred = []
    y = zip(model_list[0], model_list[1], model_list[2])
    for i in y:
        d = Counter(i)
        val = d.most_common(1)
        if val[0][1]==1:
            # pulls value from top model
            maj_pred.append(i[top_model])
        else:
            maj_pred.append(val[0][0])
    return np.array(maj_pred)

def build_model(X_train, X_test, y_train, y_test, model, grid_search=None, \
                accuracy=True, precision=True, recall=None):
    """
    Function that fits a model, predicts, and prints scores

    INPUTS:

    X_train: numpy array or pandas dataframe
        The training feature matrix.

    X_test: numpy array or pandas dataframe
        The test feature matrix.

    y_train: numpy array
        The training set labels.

    y_test: numpy array
        The test set labels.

    model: model object
        The instantiated classification model used for fitting and Scoring.

    grid_search: dictionary
        Parameter grid dictionary that will be used for grid search.

    accuracy: boolean
        If True, accruacy will be calculated and displayed.

    precision: boolean
        If True, the precision scoring metric will be calculated.

    recall: dictionary
        Integer of important class as the key and threshold for the class as a \
        value. Ex: {1:0.7}

    OUTPUT: fitted model and predictions - predicted labels or probabilities

    """
    print 'Build Model Function Running...'
    if (grid_search != None) and (recall != None):
        model = GridSearchCV(model, param_grid=grid_search).fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        y_pred = np.where( (y_pred[:,recall.keys()[0]].flatten() > \
                                recall.values()[0]), 1, 0)
        z = y_pred == 1.
        z1 = y_test[z]
        y_pos = z1 == 1.
        if accuracy == True:
            print '{} Accuracy: {}'\
                .format(model.__class__.__name__, model.score(X_test, y_test))
        if precision == True:
            print 'Precision: {}'.format( (float(sum(y_pos)) / len(y_pos)) )
        print 'Recall with threshold: {}'\
                                    .format( float(sum(y_pos)) / sum(y_test) )
    elif (grid_search != None) and (recall == None):
        model = GridSearchCV(model, param_grid=grid_search).fit(X_train, y_train)
        y_pred = model.predict(X_test)
        z = y_pred == 1.
        z1 = y_test[z]
        y_pos = z1 == 1.
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        z = y_pred == 1.
        z1 = y_test[z]
        y_pos = z1 == 1.
    if (accuracy == True) and (grid_search == None):
        print '{} Accuracy: {}'\
            .format(model.__class__.__name__, model.score(X_test, y_test))
    if (precision == True) and (grid_search == None):
        print 'Precision: {}'.format( (float(sum(y_pos)) / len(y_pos)) )
    if (recall == None) and (grid_search == None):
        print 'Recall without threshold: {}'\
                                    .format( float(sum(y_pos)) / sum(y_test) )
    return model, y_pred


if __name__ == "__main__":

    # loads train test sets dictionary
    dict_train_test = load_pickle('../data/feat_engineer_v2.pkl')

    # unpack train and test dictionary
    df_train, df_test = dict_train_test['df_train'], dict_train_test['df_test']

    # create train and test sets for modeling
    # train
    y_train = df_train.pop('Labels')
    X_train = df_train
    # test
    y_test = df_test.pop('Labels')
    X_test = df_test

    # random forest classifier
    rf = RandomForestClassifier(n_estimators=50)

    # create parameter grid for random forest
    param_grid = {"max_depth": [3, 10,100],
                  "max_features": [1, 3, 100],
                  "min_samples_split": [1, 3, 100],
                  "min_samples_leaf": [1, 3, 100],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # create fitted model and predictions
    rf_2_pkl, y_pred = build_model(X_train, X_test, y_train, y_test, rf, \
                           grid_search=param_grid, accuracy=True, \
                           precision=True, recall={1:0.05})

    # pickles the model and predictions
    save_pickle(rf_2_pkl, '../data/random_forest_v2.pkl')

    save_pickle(y_pred, '../data/predictions.pkl')

    print 'DONE!'

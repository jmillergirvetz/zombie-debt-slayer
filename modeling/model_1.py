###Module to model - LogRegCV, RF, XGBC, SVM with poly & rbf kernels###
import pandas as pd
import numpy as np
from collections import Counter

from oversample import random_over_sample

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold

from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

import cPickle as pickle


def load_pickle(picklefile):
    """
    Function that loads in pickled feature matrix
    INPUT: pickle file path
    OUTPUT: feature matrix
    """
    with open(picklefile, 'r') as f:
        pickle.load(f)

def save_pickle(data, picklefile):
    """
    Function that saves either a pickled object or pickled model
    INPUT: path to file and filename and the fitted model to pickle
    OUTPUT: pickled file to the specified path
    """
    with open(picklefile, 'w') as f:
        pickle.dump(data, f)

def train_test_sets(X, y):
    """
    Function that splits cross validation training and test sets
    INPUT: X feature matrix and y output labels
    OUTPUT: cross-validation training and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \
                                                        random_state=1)
    return X_train, X_test, y_train, y_test

# def kfold_cv(X_train, y_train):
#
#     kf = KFold(n=X_train.shape[0], n_folds=5, shuffle=False, random_state=None)
#     for train_index, test_index in kf:
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

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

def combine_model_pred(model_list):
    """
    Function that creates numpy matrix of predictions for each fitted model
    INPUT: list of numpy arrays containing predictions for each model
    OUTPUT: a numpy array containing the aggregated major predictions \
    """
    maj_pred = []
    y = zip(model_list[0], model_list[1], model_list[2])
    for i in y:
        d = Counter(i)
        val = d.most_common(1)
        if val[0][1]==1:
            # pulls value from rf because it has the highest initial score
            maj_pred.append(i[1])
        else:
            maj_pred.append(val[0][0])
    return np.array(maj_pred)

def build_model(X_train, X_test, y_train, y_test, model):
    """
    Function that returns a fitted model
    INPUT: model, X feature matrix, and y output column training and test sets
    OUTPUT: fitted model
    """
    model.fit(X_train, y_train)
    y_pred = model.pred(X_test)
    print '{} Regression Accuracy, {}'\
        .format(model.__class__.__name__, lg.score(X_orig_test, y_orig_test))
    z = y_pred == 1.
    z1 = y_test[z]
    y_tpfp_arr = z1 == 1.
    print 'Precision, {}'.format(float(sum(y_tpfp_arr)) / len(y_tpfp_arr))
    return mod


if __name__ == "__main__":

    # load feature matrix and labels
    # feat_mat = load_pickle('../data/feat_mat.pkl')
    df = load_pickle('../data/small_feat_mat_v1.pkl')

    # create X feature matrix and y output labels and drop any unwanted columns
    y = df.pop('Labels')
    X = df

    # create a random oversample of the data to train the models
    # NOTE: when testing model, use original test set
    X_resample, y_resample = random_over_sample(X, y)

    # creates original train and test sets
    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_sets(X, y)

    # create initial train and test sets with resampled balanced labels data
    X_train, X_test, y_train, y_test = train_test_sets(X_resample, y_resample)

    # instantiate, fit, and score models
    # logistic regression
    lg = LogisticRegressionCV(cv=5)

    lg_2_pkl = build_model(X_train, X_orig_test, y_train, y_orig_test, lg)
    print
    lg.fit(X_train, y_train)
    y_lg_pred = lg.predict(X_orig_test)
    print 'Logistic Regression Accuracy, ', lg.score(X_orig_test, y_orig_test)
    print
    z = y_lg_pred == 1.
    z1 = y_orig_test[z]
    y_lg_tpfp_arr = z1 == 1.
    print 'Precision, ', float(sum(y_lg_tpfp_arr)) / len(y_lg_tpfp_arr)
    print

    # random forest classifier
    rf = RandomForestClassifier()

    rf_2_pkl = build_model(X_train, X_orig_test, y_train, y_orig_test, rf)
    print
    rf.fit(X_train, y_train)
    y_rf_pred = rf.predict(X_orig_test)
    print 'Random Forest Classifer Accuracy, ', \
                                            rf.score(X_orig_test, y_orig_test)
    print
    z = y_rf_pred == 1.
    z1 = y_orig_test[z]
    y_rf_tpfp_arr = z1 == 1.
    print 'Precision, ', float(sum(y_rf_tpfp_arr)) / len(y_rf_tpfp_arr)
    print

    # gradient boosting classifier
    gbc = GradientBoostingClassifier()

    gbc_2_pkl = build_model(X_train, X_orig_test, y_train, y_orig_test, gbc)
    print
    gbc.fit(X_train, y_train)
    y_gbc_pred = gbc.predict(X_orig_test)
    print 'Gradient Boosting Classifer Accuracy, ', \
                                            gbc.score(X_orig_test, y_orig_test)
    print
    z = y_gbc_pred == 1.
    z1 = y_orig_test[z]
    y_gbc_tpfp_arr = z1 == 1.
    print 'Precision, ', float(sum(y_gbc_tpfp_arr)) / len(y_gbc_tpfp_arr)
    print

    # #############
    # # support vector machines with the following kernels:
    # # polynomial function kernel
    # svc_poly = SVC(kernel='poly').fit(X_train, y_train)
    # # g = GridSearchCV(svc_poly, {'C':np.linspace(.001, 3, 20), \
    # #                             'degree':[1,2,3,4]}, cv=10).fit(X_train, y_train)
    # print 'Support Vector Machine, Polynomial Kernel, ', \
    #                             svc_poly.score(X_orig_test, y_orig_test)
    # print
    # #print 'Grid Search Best Parameters, ', g.best_params_
    # print
    #
    # # radial basis function kernel
    # svc_rbf = SVC(kernel='rbf').fit(X_train,y_train)
    # # g = GridSearchCV(svc_rbf, {'C':np.linspace(.001, 3, 20), \
    # #                             'gamma':np.linspace(0,5,20)}, \
    # #                             cv=10).fit(X_train, y_train)
    # print 'Support Vector Machine, Polynomial Kernel, ', \
    #                             svc_rbf.score(X_orig_test, y_orig_test)
    # print
    # #print 'Grid Search Best Parameters, ', g.best_params_
    # print
    # #############

    # combines each model - LogReg, RF, XGBC SVM - and checks scores
    X_cm_pred = pred_mat(X_orig_test, [lg, rf, gbc])
    z = combine_model_pred(X_cm_pred)
    y_cm_pred = z == y_orig_test

    print 'Combined Model Accuracy, ', float(sum(y_cm_pred)) / \
                                                        y_orig_test.shape[0]
    print
    z = y_cm_pred == 1.
    z1 = y_orig_test[z]
    y_cm_tpfp_arr = z1 == 1.
    print 'Precision, ', float(sum(y_cm_tpfp_arr)) / len(y_cm_tpfp_arr)
    print

    ## Scoring Metrics:
    # Precision = TP / (TP + FP)
    # Precision: given that pred is forgiven what percentage of the time were \
    # they actually forgiven
    # Recall = TP / (TP + FN)
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)

    # pickles the models
    save_pickle(lg_2_pkl, '../data/logistic_regression.pkl')

    save_pickle(rf_2_pkl, '../data/random_forest.pkl')

    save_pickle(gbc_2_pkl, '../data/gradient_boosting_classifer.pkl')

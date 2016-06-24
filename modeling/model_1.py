import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from collections import Counter
from sklearn.metrics import confusion_matrix


def load_pickle(picklefile):
    '''
    Function that loads in pickled feature matrix
    INPUT: pickle file path
    OUTPUT: feature matrix
    '''
    with open(picklefile, 'r') as f:
        feat_mat = pickle.load(f)
    return feat_mat

def train_test_sets(X, y):
    '''
    Function that splits cross validation training and test sets
    INPUT: X feature matrix and y output labels
    OUTPUT: cross-validation training and test sets
    '''
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
    '''
    Function that creates numpy matrix of predictions for each fitted model
    INPUT: X_test set used for model predictions and a list of fitted models
    OUTPUT: list of numpy arrays containing predictions for each model
    '''
    L = []
    for model in model_list:
        L.append(model.predict(X_test))
    return L

def combine_model_pred(model_list):
    '''
    Function that creates numpy matrix of predictions for each fitted model
    INPUT: list of numpy arrays containing predictions for each model
    OUTPUT: a numpy array containing the aggregated major predictions \

    '''
    maj_pred = []
    y = zip(model_list[0], model_list[1], model_list[2])
    print y[:10]
    for i in y:
        d = Counter(i)
        val = d.most_common(1)
        if val[0][1]==1:
            maj_pred.append(i[2]) # pulls value from xgbc
        else:
            maj_pred.append(val[0][0])
    return np.array(maj_pred)

def std_confusion_matrix(y_true, y_predict):
    '''
    Function that creates standard confusion matrix of tp, fp, fn, tn
    INPUT: a numpy arrays of true and predicted output values
    OUTPUT: a numpy array of a standard confusion matrix
    '''
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_predict)
    return np.array([[tp, fp], [fn, tn]])

if __name__ == "__main__":

    # load feature matrix and labels
    # feat_mat = load_pickle('../data/feat_mat.pkl')
    df = load_pickle('../data/small_feat_mat.pkl')

    # create X feature matrix and y output labels and drop any unwanted columns
    y = df.pop('Labels')
    df = df.drop('ZIP code', axis=1)
    X = df
    print 'X ', X.shape


    # create initial cross-validation train and test sets
    X_train, X_test, y_train, y_test = train_test_sets(X, y)
    print 'X_train ', X_train.shape
    print
    print 'y_train ', y_train.shape
    print
    print 'x_test', X_test.shape

    # instantiate, fit, and score models
    # logistic regression
    lg = LogisticRegressionCV(cv=5)
    lg.fit(X_train, y_train)
    y_lg_pred = lg.predict(X_test)
    print 'y_test ', y_test.shape
    print 'log reg', lg.score(X_test, y_test)
    print

    # random forest classifier
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_rf_pred = rf.predict(X_test)
    print 'random forest', rf.score(X_test, y_test)
    print

    # gradient boosting classifier
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    y_gb_pred = gb.predict(X_test)
    print 'gradient boosting', gb.score(X_test, y_test)
    print

    predictions = pred_mat(X_test, [lg, rf, gb])
    pred_arr = combine_model_pred(predictions)

    # compares the actual test values with the predicted values to determine if
    # aggregating the results improves the accuracy of the model
    # Precision = TP / (TP+FP)
    # Recall = TP / (TP+FN)
    # Accuracy = (tp + tn) / (tp + tn + fp + fn)

    X_test_new = y_test==pred_arr

    print X_test_new.sum() / float(y_test.shape[0])
    # print 'y_test', y_test.shape
    # print 'y_pred', pred_arr.shape
    # # creates confusion matrix
    # print std_confusion_matrix(y_test, pred_arr)
    # # OUTPUT: np.array([[tp, fp], [fn, tn]])
    # print cm_arr
    # print "TP", cm_arr[0][0]
    # print "FP", cm_arr[0][1]
    # print "TN", cm_arr[1][1]
    # print "FN", cm_arr[1][0]
    # print "Precision = TP / (TP+FP)", cm_arr[0][0] / (cm_arr[0][0] + cm_arr[0][1])
    # print "Recall = TP / (TP+FN)", cm_arr[0][0] / (cm_arr[0][0] + cm_arr[1][0])
    # print "Accuracy = (tp + tn) / (tp + tn + fp + fn)", \
    #                                 (cm_arr[0][0] + cm_arr[1][1]) / cm_arr.sum()

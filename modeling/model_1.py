import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold


def load_pickle(picklefile):
    '''
    Function that loads in pickled feature matrix
    INPUT: pickle file path
    OUTPUT: feature matrix
    '''
    with open(picklefile, 'r') as f:
        feat_mat = pickle.load(f)
    return feat_mat

def cross_validation(X, y):
    '''
    Function that splits cross validation training and test sets
    INPUT: X feature matrix and y output labels
    OUTPUT: cross-validation training and test sets
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \
                                                        random_state=1)
    return X_train, X_test, y_train, y_test

def kfold_cv(X_train, y_train):

    kf = KFold(n=X_train.shape[0], n_folds=5, shuffle=False, random_state=None)
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]


if __name__ == "__main__":

    # load feature matrix and labels
    # feat_mat = load_pickle('../data/feat_mat.pkl')
    df = load_pickle('../data/small_feat_mat.pkl')

    # create X feature matrix and y output labels and drop any unwanted columns
    y = df.pop('Labels')
    df = df.drop('ZIP code', axis=1)
    X = df
    print y
    print
    print 'X ', X.shape
    #print np.array(X.info()).T



    # create initial cross-validation train and test sets
    X_train, X_test, y_train, y_test = cross_validation(X, y)
    print 'X_train ', X_train.shape
    print
    print 'y_train ', y_train.shape
    print
    print 'x_test', X_test.shape
    # instantiate, fit, and score models
    # logistic regression
    lg = LogisticRegressionCV(cv=5)
    lg.fit(X_train, y_train)
    #y_lg_pred = lg.predict(X_test)
    print 'y_test ', y_test.shape
    #print 'y_pred ', y_lg_pred.shape
    print 'log reg', lg.score(X_test, y_test)
    print

    # random forest classifier
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    #y_rf_pred = rf.predict(X_test)
    print 'random forest', rf.score(X_test, y_test)
    print

    # gradient boosting classifier
    gb = GradientBoostingClassifier()
    gb.fit(X_train, y_train)
    #y_gb_pred = gb.predict(X_test)
    print 'gradient boosting', gb.score(X_test, y_test)
    print

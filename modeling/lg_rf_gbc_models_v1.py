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

def build_model(X_train, X_test, y_train, y_test, model, accuracy=True, \
                precision=True, recall=None):

    """
    Function that returns a fitted model and scores it
    INPUT: model, X feature matrix, and y output column training and test sets
    OUTPUT: fitted model and print scores
    """
    print 'Model Function'
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    z = y_pred == 1.
    z1 = y_test[z]
    y_pos = z1 == 1.
    if accuracy == True:
        print '{} Accuracy: {}'\
            .format(model.__class__.__name__, model.score(X_test, y_test))
    if precision == True:
        print 'Precision: {}'.format( (float(sum(y_pos)) / len(y_pos)) )
    if recall != None:
        y_pred = model.predict_proba(X_test)
        y_pred = y_pred.where( (y_pred[:,1] > (0.33 * recall)), 1, 0)
        z = y_pred == 1.
        z1 = y_test[z]
        y_pos = z1 == 1.
        print 'Recall with threshold: {}'\
                                    .format( float(sum(y_pos)) / sum(y_test) )
    else:
        print 'Recall without threshold: {}'\
                                    .format( float(sum(y_pos)) / sum(y_test) )
    return model



if __name__ == "__main__":

    # loads train test sets dictionary
    dict_train_test = load_pickle('../data/feat_engineer.pkl')

    # unpack train and test dictionary
    df_train, df_test = dict_train_test['df_train'], dict_train_test['df_test']

    ############
    # TEST text columns verse not
    # df_train = df_train.drop(['Narrative length', 'Polite', 'Punctuation', 'First person count'], axis=1)
    # df_test = df_test.drop(['Narrative length', 'Polite', 'Punctuation', 'First person count'], axis=1)
    ############

    # create train and test sets for modeling
    # train
    y_train = df_train.pop('Labels')
    X_train = df_train
    # test
    y_test = df_test.pop('Labels')
    X_test = df_test

    # instantiate, fit, and score models
    # logistic regression
    # lg = LogisticRegressionCV(cv=5)
    #
    # # lg_2_pkl = build_model(X_train, X_test, y_train, y_test, lg, recall=0.7)
    # print
    # lg.fit(X_train, y_train)
    # y_logit_pred = lg.predict(X_test)
    # y_logit_pred_prob = lg.predict_proba(X_test)
    #
    # print 'Logistic Regression Accuracy: ', lg.score(X_test, y_test)
    # print
    # z = y_logit_pred == 1.
    # z1 = y_test[z]
    # y_logit_pos = z1 == 1.
    # print 'Precision: ', float(sum(y_logit_pos)) / len(y_logit_pos)
    # print
    # print 'Recall: ', float(sum(y_logit_pos)) / sum(y_test)
    # print

    # random forest classifier
    rf = RandomForestClassifier(n_estimators=20)

    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [1, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    g = GridSearchCV(rf, param_grid=param_grid).fit(X_train, y_train)
    print "Grid Scores:", g.grid_scores_

    rf_2_pkl = build_model(X_train, X_test, y_train, y_test, rf)


    # print
    # rf.fit(X_train, y_train)
    # y_rf_pred = rf.predict(X_test)
    # print 'Random Forest Classifier Accuracy: ', \
    #                                         rf.score(X_test, y_test)
    # print
    # z = y_rf_pred == 1.
    # z1 = y_test[z]
    # y_rf_pos = z1 == 1.
    # print 'Precision: ', float(sum(y_rf_pos)) / len(y_rf_pos)
    # print
    # print 'Recall: ', float(sum(y_rf_pos)) / sum(y_test)
    # print

    # gradient boosting classifier
    gbc = GradientBoostingClassifier()

    #gbc_2_pkl = build_model(X_train, X_test, y_train, y_test, gbc)
    print
    gbc.fit(X_train, y_train)
    y_gbc_pred = gbc.predict(X_test)
    print 'Gradient Boosting Classifier Accuracy: ', \
                                            gbc.score(X_test, y_test)
    print
    z = y_gbc_pred == 1.
    z1 = y_test[z]
    y_gbc_pos = z1 == 1.
    print 'Precision: ', float(sum(y_gbc_pos)) / len(y_gbc_pos)
    print
    print 'Recall: ', float(sum(y_gbc_pos)) / sum(y_test)
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
    X_com_pred = pred_mat(X_test, [lg, rf, gbc])
    z = combine_model_pred(X_com_pred)
    y_com_pred = z == y_test

    print 'Combined Model Accuracy: ', float(sum(y_com_pred)) / \
                                                        y_test.shape[0]
    print
    z = y_com_pred == 1.
    z1 = y_test[z]
    y_com_pos = z1 == 1.
    print 'Precision: ', float(sum(y_com_pos)) / len(y_com_pos)
    print
    print 'Recall: ', float(sum(y_com_pos)) / sum(y_test)
    print

    print 'DONE!'
    ## Scoring Metrics:
    # Precision = TP / (TP + FP)
    # Precision: given that pred is forgiven what percentage of the time were \
    # they actually forgiven
    # Recall = TP / (TP + FN)
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)

    # pickles the models
    save_pickle(lg, '../data/logistic_regression.pkl')

    save_pickle(rf, '../data/random_forest.pkl')

    save_pickle(gbc, '../data/gradient_boosting_classifer.pkl')

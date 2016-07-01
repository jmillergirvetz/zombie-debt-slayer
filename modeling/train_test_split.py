import pandas as pd
import numpy as np
import unidecode

from oversample import random_over_sample
from sklearn.cross_validation import train_test_split

from lg_rf_gbc_models import load_pickle, save_pickle
import cPickle as pickle


def train_test_sets(X, y):
    """
    Function that splits cross validation training and test sets
    INPUT: X feature matrix and y output labels
    OUTPUT: cross-validation training and test sets
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \
                                                        random_state=1)
    return X_train, X_test, y_train, y_test

def merge_dataframes(df, df1):
    cols = list(set(df1.columns) - {'Company', 'Consumer complaint narrative', \
                                    'Date received'})
    result = pd.merge(df, df1, how='left', on=cols)
    return result

def convert_text(text):
    if (text == ''):
        print "FOUND MISSING TEXT"
        return "--MISSING INFO--"
    else:
        return unidecode.unidecode_expect_nonascii(text)


if __name__ == "__main__":

    # load feature matrix and labels
    df = load_pickle('../data/init_feat_mat.pkl')
    df['index'] = df.index
    df = df.dropna(subset=np.array(['Consumer complaint narrative']), axis=0)

    # converts strings to unicode
    df['Company'] = df['Company'].apply(lambda x: convert_text(x))
    df['Consumer complaint narrative'] = df['Consumer complaint narrative']\
                                         .apply(lambda x: convert_text(x))
    # there were no --MISSING INFO-- values (see convert_text function)

    # create X feature matrix and y output labels and drop any unwanted columns
    y = df.pop('Labels')
    X = df
    # create initial train and test sets with resampled balanced labels data
    X_train, X_test, y_train, y_test = train_test_sets(X, y)

    # create new dataframe with X_train
    X_train_drop = X_train.drop(['Company', 'Consumer complaint narrative', \
                                 'Date received'], axis=1)

    # create array of columns names from the dropped column dataframe
    X_drop_cols = np.array(X_train_drop.columns)

    # create a random oversample of the data to train the models
    # NOTE: when testing model, use original test set
    X_resample, y_resample_train = random_over_sample(X_train_drop.values, \
                                                                y_train.values)

    # convert resampled test set to dataframe
    X_resample = pd.DataFrame(data=X_resample, columns=X_drop_cols)

    # merge oversampled dataframe with ints and floats with dataframe with str
    X_resample_train = merge_dataframes(X_resample, X_train)

    # drop temp index columns used to map a correct merge
    X_resample_train = X_resample_train.drop(['index'], axis=1)
    X_test = X_test.drop(['index'], axis=1)

    # create dictionary of columns, mapping columns to indeces
    X_dict_cols = dict( zip(X_resample_train.columns, \
                            range(X_resample_train.shape[1])) )

    # create train test sets dictionary
    d_train_test = {'X_col_dic':X_dict_cols, 'X_train':X_resample_train, \
                    'X_test':X_test, 'y_train':y_resample_train, \
                    'y_test':y_test}

    # save pickled dictionaries with original and oversampled data sets
    save_pickle(d_train_test, '../data/train_test_resamp.pkl')

import pandas as pd
import numpy as np
from datetime import datetime


def drop_columns(df, columns):
    '''
    Function that drops specified columns
    INPUT: dataframe and list of columns
    OUTPUT: cleaned dataframe
    '''
    return df.drop(columns, axis=1)

def remove_nan(df, columns):
    '''
    Function that drops specified rows of dataframe that have NaN values
    INPUT: dataframe and list of columns that specify which NaN values to drop
    OUTPUT: cleaned dataframe
    '''
    return df.dropna(axis=0, subset=np.array(columns))

def get_dummies(df, columns=None):
    '''
    Function that creates dummie variables for categorical variables
    INPUT: dataframe
    OUTPUT: dataframe with dummy variables
    '''
    return pd.get_dummies(df, dummy_na=True, columns=columns)


if __name__ == "__main__":

    # specify data types on import
    col_data_types = {'Consumer complaint narrative':np.str_, \
                      'Company public response':np.str_, \
                      'Consumer consent provided?':np.str_, \
                      'Date received':datetime, \
                      'Date sent to company':datetime}
    # load in data
    df = pd.read_csv('../data/Consumer_Complaints.csv', dtype=col_data_types)

    # drop columns that are represented by other columns
    df = drop_columns(df, ['Sub-product', 'Sub-issue', \
                    'Consumer complaint narrative', \
                    'Company public response'])

    # removes rows with NaN values for state and zip. There are < 5000 rows
    df = remove_nan(df, ['State', 'ZIP code', 'Submitted via'])

    # creates boolean categorical variables
    df['Older American'] = pd.notnull(df['Tags'].str.contains('Older'))
    df['Service member'] = df['Tags'].str.contains('Service').replace(np.nan, False)
    df['Disputed?'] = df['Consumer disputed?'].str.contains('Yes').replace(np.nan, False)
    df['Narrative consent provided'] = df['Consumer consent provided?']=='Consent provided'
    df['Timely response?'] = df['Timely response?']=='Yes'

    # drops old columns that have been converted to booleans
    df = drop_columns(df, ['Tags', 'Consumer disputed?', 'Consumer consent provided?', 'Timely response?'])

    # creates list of date, zip, and claim id columns
    date_id_zip = ['Date received', 'ZIP code', 'Date sent to company', 'Complaint ID']

    # creates list of boolean categorical variables
    bool_list = ['Narrative consent provided', 'Timely response?', 'Disputed?', \
                 'Service member', 'Older American']

    # creates list categorical variables to be dummified
    categorical_var = list(set(df.columns) - set(date_id_zip + bool_list))

    # dummifies categorical variables
    df = get_dummies(df, categorical_var)

    print df.shape
    print len(df.columns)

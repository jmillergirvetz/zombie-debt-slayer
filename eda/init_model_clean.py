###Module to preprocess data###
import pandas as pd
import numpy as np

import cPickle as pickle
import json

def drop_columns(df, columns):
    """
    Function that drops specified columns
    INPUT: dataframe and list of columns
    OUTPUT: cleaned dataframe
    """
    return df.drop(columns, axis=1)

def remove_nan(df, columns):
    """
    Function that drops specified rows of dataframe that have NaN values
    INPUT: dataframe and list of columns that specify which NaN values to drop
    OUTPUT: cleaned dataframe
    """
    return df.dropna(axis=0, subset=np.array(columns))

def get_dummies(df, columns=None):
    """
    Function that creates dummie variables for categorical variables
    INPUT: dataframe
    OUTPUT: dataframe with dummy variables
    """
    return pd.get_dummies(df, dummy_na=True, columns=columns)

def create_year_only(df, date_col='Date received'):
    # function thats create a 'Year only' column
    df[date_col] = df[date_col].apply(lambda x: str(x))
    df[date_col] = pd.to_datetime(df[date_col])
    df['Year only'] = pd.DatetimeIndex(df[date_col]).year
    return df

def save_pickle(data, picklefile):
    """
    Function that saves pickled data
    INPUT: feature matrix with class labels
    OUTPUT: pickled feature matrix with class labels
    """
    with open(picklefile, 'wb') as f:
        pickle.dump(data, f)

def read_json(jsonfile):
    """
    Function that reads json file
    INPUT: json file with path
    OUTPUT: json dictionary
    """
    with open(jsonfile, 'rb') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":

    # specify data types on import
    col_data_types = {'Consumer complaint narrative':np.str_, \
                      'Company public response':np.str_, \
                      'Consumer consent provided?':np.str_}
    # load in data
    df = pd.read_csv('../data/Consumer_Complaints.csv', dtype=col_data_types)

    # load in dictionary of states and 2012 presidential election results
    state_dict = read_json('../data/red_blue_states.json')

    # drop columns that are represented by other columns
    df = drop_columns(df, ['Sub-product', 'Sub-issue', \
                            'Company public response', \
                            'Date sent to company', \
                            'Complaint ID'])

    # removes rows with NaN values ~ 550,000 rows
    df = remove_nan(df, ['State', 'ZIP code', 'Submitted via', \
                        'Consumer complaint narrative'])

    # creates boolean categorical variables
    df['Older American'] = pd.notnull(df['Tags'].str.contains('Older'))
    df['Service member'] = df['Tags'].str.contains('Service').replace(np.nan, False)
    df['Disputed?'] = df['Consumer disputed?'].str.contains('Yes').replace(np.nan, False)
    df['Narrative consent provided'] = df['Consumer consent provided?']=='Consent provided'
    df['Timely response?'] = df['Timely response?']=='Yes'
    df['ZIP code'] = df['ZIP code'].apply(lambda x: str(x)[:3])
    df['State'] = df['State'].replace(state_dict)
    df['Labels'] = df['Company response to consumer']\
                            .replace({'Closed with non-monetary relief':0., \
                                    'Closed with explanation':2.,\
                                    'Closed with monetary relief':1., \
                                    'Closed':2., \
                                    'Untimely response':2., \
                                    'In progress':2., \
                                    'Closed with relief':1., \
                                    'Closed without relief':0.})

    # drops old columns that have been converted to booleans and output labels
    df = drop_columns(df, ['Tags', 'Consumer disputed?', \
                            'Consumer consent provided?', \
                            'Company response to consumer'])

    # creates set of boolean categorical variables
    bool_set = {'Narrative consent provided', 'Timely response?', 'Disputed?', \
                'Service member', 'Older American'}

    # creates set of holdout features and output to be processed later
    comp_date_narr_label = {'Company', 'Date received', \
                            'Consumer complaint narrative', 'Labels'}

    # creates list of categorical variables to be dummified
    categorical_var = list(set(df.columns) - bool_set - comp_date_narr_label)

    # dummifies categorical variables
    df_dummy_categories = get_dummies(df, categorical_var)

    # creates datetime64 year only column
    df = create_year_only(df_dummy_categories, date_col='Date received')

    # pickle cleaned dataframe
    save_pickle(df, '../data/init_feat_mat.pkl')

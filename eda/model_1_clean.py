###Module to preprocess data###
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

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

def save_model(data, picklefile):
    """
    Function that saves pickled data
    INPUT: feature matrix with class labels
    OUTPUT: pickled feature matrix with class labels
    """
    with open(picklefile, 'w') as f:
        pickle.dump(data, f)

def read_json(jsonfile):
    """
    Function that reads json file
    INPUT: json file with path
    OUTPUT: json dictionary
    """
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    return data

if __name__ == "__main__":

    # specify data types on import
    col_data_types = {'Consumer complaint narrative':np.str_, \
                      'Company public response':np.str_, \
                      'Consumer consent provided?':np.str_}
    # load in data
    df = pd.read_csv('../data/Consumer_Complaints.csv', dtype=col_data_types)

    state_dict = read_json('../data/red_blue_states.json')

    # drop columns that are represented by other columns
    # NOTE: some of these may need to be included in feature matrix |
    # The 'Company' feature is too large after creating dummy variables
    # Need to solve the curse of dimensionality and sparse matrix problems
    df = drop_columns(df, ['Sub-product', 'Sub-issue', \
                        'Consumer complaint narrative', \
                        'Company public response', \
                        'Company', 'Date sent to company', \
                        'Date received', 'Complaint ID'])

    # removes rows with NaN values for state and zip | there are < 5000 NaN rows
    df = remove_nan(df, ['State', 'ZIP code', 'Submitted via'])

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

    # # encode class labels: used for encoding multi-classes if applicable
    # le = LabelEncoder()
    # label_arr = le.fit_transform(np.array(df['Company response to consumer']))
    # df['Labels'] = pd.DataFrame(label_arr)

    # drops old columns that have been converted to booleans and output labels
    df = drop_columns(df, ['Tags', 'Consumer disputed?', \
                            'Consumer consent provided?', \
                            'Company response to consumer'])

    # creates list of boolean categorical variables
    bool_set = {'Narrative consent provided', 'Timely response?', 'Disputed?', \
                'Service member', 'Older American'}

    # creates list of categorical variables to be dummified
    categorical_var = list(set(df.columns) - bool_set - {'Labels'})

    # dummifies categorical variables
    df_dummy_categories = get_dummies(df, categorical_var)

    # checks and removes "Lables" rows with null or corrupted data types
    print 'Check: "Label" # null values before removal: ', \
                                df_dummy_categories['Labels'].isnull().sum()

    df_dummy_categories = remove_nan(df_dummy_categories, ['Labels'])

    print
    print 'Check: "Label" # null values after removal: ', \
                                df_dummy_categories['Labels'].isnull().sum()

    print
    print 'Cleaned dataframes dimensions:', df_dummy_categories.shape

    # pickle cleaned dataframe
    # save_model(df_feat_mat, '../data/feat_mat.pkl')

    save_model(df_dummy_categories.sample(n=10000, random_state=1), '../data/small_feat_mat.pkl')

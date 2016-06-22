import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.preprocessing import LabelEncoder


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

def save_model(data, picklefile):
    '''
    Function that saves pickled data
    INPUT: feature matrix with class labels
    OUTPUT: pickled feature matrix with class labels
    '''
    with open(picklefile, 'w') as f:
        pickle.dump(data, f)


if __name__ == "__main__":

    # specify data types on import
    col_data_types = {'Consumer complaint narrative':np.str_, \
                      'Company public response':np.str_, \
                      'Consumer consent provided?':np.str_}
    # load in data
    df = pd.read_csv('../data/Consumer_Complaints.csv', dtype=col_data_types)

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

    # encode class labels
    le = LabelEncoder()
    label_arr = le.fit_transform(np.array(df['Company response to consumer']))

    # drops old columns that have been converted to booleans and output labels
    df = drop_columns(df, ['Tags', 'Consumer disputed?', \
                            'Consumer consent provided?', \
                            'Company response to consumer'])

    # creates list of boolean categorical variables
    zip_bool_list = ['ZIP code', 'Narrative consent provided', \
                    'Timely response?', 'Disputed?', \
                    'Service member', 'Older American']

    # creates a dataframe of cleaned date, claim id, zip, and boolean features
    df_zip_bool = df[zip_bool_list]

    # creates list of categorical variables to be dummified
    categorical_var = list(set(df.columns) - set(zip_bool_list))

    # dummifies categorical variables
    df_dummy_categories = get_dummies(df, categorical_var)


    #####print block for debugging
    # print df_date_zip_id_bool.values.shape
    # print df_dummy_categories.values.shape
    # print label_arr.shape
    # print label_arr.reshape((label_arr.shape[0],1)).shape
    # print df['Company response to consumer'].values\
    #     .reshape((df['Company response to consumer'].shape[0], 1)).shape
    ######


    # creates feature matrix with classifcation labels
    feat_mat = np.concatenate((df_zip_bool.values, \
                            df_dummy_categories.values, \
                            label_arr.reshape((label_arr.shape[0],1))), axis=1)

    save_model(feat_mat, '../data/feat_mat.pkl')

    save_model(feat_mat[:1000,:], '../data/small_feat_mat.pkl')

###Module that engineers number complaints, length of narratives, formatting###
import pandas as pd
import numpy as np
from datetime import datetime

from textblob import TextBlob

from init_model_clean import drop_columns, save_pickle

import cPickle as pickle


rude_vocab = \
  set(['I will not', 'I won\'t', 'I am not going', 'I cant', 'I can\'t', \
  'I can not', 'corrupt', 'without hearing', 'illegal', 'make me', \
  'harassing', 'harass', 'harassment', 'ridiculous', 'nutshell', \
  'We certainly know', 'poor credit', 'credit being poor', 'delete', \
  'statute of limitations', 'impossible to deal', 'without telling me', \
  'I \'m not asking you for advice', 'I ca n\'t', 'deleted', ''])

polite_vocab = \
  set(['implying', 'allowed', 'protested', 'dismissed', 'technicality', 'innacurate', \
  'law', 'incorrect', 'reporting', 'contract', 'agreed', 'Passport', 'agreement', \
  'assured', 'reluctantly', 'paperwork', 'attempted', 'terms', 'resolution', \
  'initiate', 'detailed', 'letter', 'receipt', 'dispute', 'proof', 'admitted', \
  'verification', 'verify', 'threatening', 'notary', 'notarized', 'ordered' \
  'requested', 'responsibility', 'hardship', 'assured', 'official', 'forbearance', \
  'exploitative', 'unauthorized', 'passed away', 'single mother', 'single father', \
  'case number', 'disputed', 'grateful', 'opportunity', 'research', 'shocking', \
  'unethical', 'resorted', 'report', 'reported', 'target', 'despite', 'cosigner', \
  'taken advantage', 'took advantage', 'student', 'graduate', 'consolidate', \
  'burden', 'Act', 'request', 'validation', 'contractual', 'obligation', \
  'invalidated', 'constitute', 'condescending', 'crying', 'tears', 'threats', \
  'activist', 'threaten', 'harassment', 'endure', 'ineffectual', 'letter', 'vicious', \
  'never received', 'followed up', 'rectified', 'confirmation', 'confirming', \
  'on file', 'notes', 'evidence', 'predatory', 'violation', 'section', \
  'documentation', 'no avail', 'tried', 'requisition', 'statements', 'contracts', \
  'pertinent', 'validity', 'qualifications', 'misleading', 'fake', 'false', \
  'paperwork', 'communication', 'instructed', 'inconvenient', 'inconvenience', \
  'correspondence', 'laboriously', 'proving', 'blemish', 'fraudulent', 'never'])

def load_pickle(picklefile):
    """
    Function that loads in pickled feature matrix
    INPUT: pickle file path
    OUTPUT: feature matrix
    """
    with open(picklefile, 'rb') as f:
        data = pickle.load(f)
    return data

def create_narrative_len(df, text_col='Consumer complaint narrative'):
    """
    Function that creates a column of the narrative lengths
    INPUT: dataframe with narratives
    OUTPUT: dataframe with a narrative length column
    """
    df['Narrative length'] = df[text_col].apply(lambda x: len(x))
    return df

def create_formatted(df, text_col='Consumer complaint narrative'):
    """
    Function that creates a column a determines if the text is formatted
    INPUT: dataframe with narratives
    OUTPUT: dataframe with a boolean formatted column
    """
    df['Paragraphs'] = df[text_col].apply(lambda x: 1 if '\n' in x else 0)
    return df

def create_complaints(df, columns=['Company', 'Year only'], orig_date_col='Date received'):
    """
    Function that creates column of number of complaints per year
    INPUT: cleaned dataframe
    OUTPUT: dataframe with number of complaints per year column
    """
    num_complaints = 'Num complaints per year'
    df1 = df
    df1 = df1.groupby(columns)[orig_date_col].count().reset_index()
    df1 = df1.rename(index=str, columns={orig_date_col: num_complaints})
    df2 = pd.merge(df, df1, how='left', on=['Company', 'Year only'])
    return df2

def count_polite_rude(text, polite=True):
    count = 0
    if polite == True:
        for word in polite_vocab:
            if word in text:
                count += 1
        return count
    elif polite == False:
        for word in rude_vocab:
            if word in text:
                count += 1
        return count
    else:
        print "Select paramter: polite = True | False"

def count_punctuation(text):
    count = 0
    punctuation = ['.', '!', '?', ',', ';', ':']
    for symb in punctuation:
        if symb in text:
            count += 1
    return count

def count_I(text):
    count = 0
    if ' I ' in text:
        count += 1
    return count

def text_polarity(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


if __name__ == '__main__':

    # loads cleaned dataframe with dates, narratives,
    dict_train_test = load_pickle('../data/train_test_resamp.pkl')

    # loads oversampled train sets and original test set
    X_dict, X_train, X_test, y_train, y_test = dict_train_test['X_col_dic'], \
                                   dict_train_test['X_train'], \
                                   dict_train_test['X_test'], \
                                   dict_train_test['y_train'], \
                                   dict_train_test['y_test']

    # create dataframe for final analysis
    # train
    df_train = pd.concat([X_train, pd.DataFrame(y_train, \
                                        columns=np.array(['Labels']))], axis=1)
    df_train = df_train.dropna(axis=0, how='any', subset=np.array(['Labels']))
    # test
    df_test = pd.concat([X_test, pd.DataFrame(y_test, \
                                        columns=np.array(['Labels']))], axis=1)
    df_test = df_test.dropna(axis=0, how='any', subset=np.array(['Labels']))

    # create a dataframe with length of narratives column
    # train
    df_train = create_narrative_len(df_train)
    # test
    df_test = create_narrative_len(df_test)

    # create a dataframe with a boolean formatted column
    # train
    df_train = create_formatted(df_train)
    # test
    df_test = create_formatted(df_test)

    # create number of complaints per year column
    # train
    df_train = create_complaints(df_train)
    # test
    df_test = create_complaints(df_test)

    # calculate politness metric and create polite column
    # train
    df_train['Polite'] = df_train['Consumer complaint narrative']\
                                        .apply(lambda x: count_polite_rude(x))
    # test
    df_test['Polite'] = df_test['Consumer complaint narrative']\
                                        .apply(lambda x: count_polite_rude(x))

    # calculate punctuation
    # train
    df_train['Punctuation'] = df_train['Consumer complaint narrative']\
                                        .apply(lambda x: count_punctuation(x))
    # test
    df_test['Punctuation'] = df_test['Consumer complaint narrative']\
                                        .apply(lambda x: count_punctuation(x))

    # count the number of times " I " appears in text
    # train
    df_train['First person count'] = df_train['Consumer complaint narrative']\
                                        .apply(lambda x: count_I(x))
    # test
    df_test['First person count'] = df_test['Consumer complaint narrative']\
                                        .apply(lambda x: count_I(x))

    # calculate text blob polarity
    # train
    df_train['Polarity'] = df_train['Consumer complaint narrative']\
                                    .apply(lambda x: float(text_polarity(x)))
    # test
    df_test['Polarity'] = df_test['Consumer complaint narrative']\
                                    .apply(lambda x: float(text_polarity(x)))

    # drop old date columns used for preprocessing
    # train
    df_train = drop_columns(df_train, columns=['Date received', 'Year only', \
                                    'Consumer complaint narrative', 'Company'])
    # test
    df_test = drop_columns(df_test, columns=['Date received', 'Year only', \
                                    'Consumer complaint narrative', 'Company'])

    # create dictionary to store oversampled train and original test dataframes
    dict_train_test = {'df_train':df_train, 'df_test':df_test}

    # save pickled dataframe
    save_pickle(dict_train_test, '../data/feat_engineer_v2.pkl')

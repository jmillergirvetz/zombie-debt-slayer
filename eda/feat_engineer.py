###Module that engineers number complaints, length of narratives, formatting###
import pandas as pd
import numpy as np
from datetime import datetime

from textblob import TextBlob

from init_model_clean import drop_columns, save_pickle

import cPickle as pickle


polite_vocab = \
  set(['allegedly', 'perhaps', 'I love', 'I feel as if', 'email them for help', \
       'I would like my debt cleared', 'I have contacted them many times', \
       'we have been very loyal customers', 'we trusted them', \
       'please help us', 'this is a lot of money', \
       'would not have agreed had we known', \
       'the gentleman who helped me', 'thank you', \
       'I submitted a complaint', 'I requested', 'I explained', \
       'It was my understanding', 'not acceptable', 'please', 'reach out', \
       'please help me', 'hello', 'I asked', 'work with me', \
       'I am not refusing', 'tried to resolve', 'appropriate', \
       'reached out', 'availability to assist','responsibly', \
       'completely understand', 'pleasant', 'Thank you', 'documentation'])

formal_vocab = \
  set(['benefits', 'closed unexpectedly', 'I was required to pay', \
       'I was not told', 'I am on a fixed budget', 'due to merchant error', \
       'not get an accurate understanding', 'I called them multiple times', \
       'overdraft fee', ', I was advised', 'excessive', 'subsequent', \
       'customer courtesy', 'compromised', 'to no avail', 'disputed', \
       'discreprency', 'reinstatement', 'forbearance', 'to assure me as such',
       'reiterated', 'obligations', 'proprietary', 'substantial', 'forfeited'])

rude_vocab = \
  set(['violated', 'serious breach', 'demanding', 'fraudulent', 'robbery', \
       'unfair business practice', 'this is ridiculous','flabbergasted', \
       'was not helpful', 'made up crazy story', 'getting the run around', \
       'victim', 'fraud', 'MORE', 'lie', 'false', 'conflicting information', \
       'failed', 'attacking', 'rude', 'arguing', 'abusing', 'disputing', \
       'demanding', 'shocking', 'trick', 'irresponsibly', 'incompetent', \
       'poor practice', 'ignored', 'fraudulently', 'extorted', 'extortionist', \
       'extorting', 'harassed', 'serious breach', 'horror', 'illegal', \
       'I DID NOT GIVE MY PERMISSION', 'incompitent', 'incompetent', \
       'misrepresented', 'refused', 'liar', 'lier','lied', 'lying', 'insane', \
       'negligent', 'foolishness', 'damaging', 'derogatory', 'worse at this', \
       'unethical', 'harassing', 'negative', 'rude', 'interrupted', 'hung up', \
       'sue', 'unfairly', 'violated', 'deprive', 'refusing', 'misleading', \
       'commingled', 'direct opposition', 'manipulating',  'threatened', \
       'insulting', 'unacceptable', 'aggravated', 'hostile', 'dysfunctional', \
       'untrustworthy', ';wishing scam', 'suffered', 'disappointing', \
       'hung up', 'steal', 'collusion', 'wastes time',  'fake'])

informal_vocab = \
  set(['I totally don\'t get', 'ASAP', 'don\'t know what to do', \
  'not allowing me to fix it', 'will not update', 'can\'t access', \
  'it isn\'t fair', 'freaked out', 'no reason', 'bugging us', 'so called', \
  'lip service', 'fruitless', 'place on long holds', 'shocked', 'surprise', \
  'fix', 'fixed', 'come to find out', 'messed up', 'Never lived', 'Never had', \
  'can\'t believe', 'NOT', 'NEVER', 'worried', 'i don\'t get any assistance',  \
  'fair', 'wrong', 'hung up', 'strikes and their out', 'run around', \
  'supposed to', 'quick fix', 'back to square one', 'screwed up', 'mess', \
  'didn\'t give me anything', 'want this to stop', 'burden', \
  'couldn\'t get to my funds', 'NONO', 'in my book',  'bogged down', \
  'stressed out', 'last ditch effort', 'screwed', \
  'Please, somebody out there help me!', 'does not give you anything', 'sucks', \
  'headaches','run around', 'don\'t live up to', 'what is this??', 'annoying', \
  'no desire to speak', 'bewilderment', 'going in circles', 'Its NOT right', \
  'Something need to be done', 'bogus charges', 'don\'t feel comfortable', \
  'Sadly', 'devastated'])


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

def count_polite_rude(text):
    count = 0
    for word in polite_vocab:
        if word in text:
            count += 1
    for word in rude_vocab:
        if word in text:
            count -= 1
    return count

def count_formal_informal(text):
    count = 0
    for word in formal_vocab:
        if word in text:
            count += 1
    for word in informal_vocab:
        if word in text:
            count -= 1
    return count

def count_punctuation(text):
    count = 0
    punctuation = ['.', '!', '?', ',', ';', ':']
    for symb in punctuation:
        if symb in text:
            count += 1
    return count

def count_I(text):
    count = 0
    first_person = ['I', 'me', 'Me', 'my', 'My', 'mine', 'Mine']
    for word in first_person:
        if word in text:
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
                        .apply(lambda x: count_polite_rude(x) / float(len(x)))
    # test
    df_test['Polite'] = df_test['Consumer complaint narrative']\
                        .apply(lambda x: count_polite_rude(x) / float(len(x)))

    # calculate politness metric and create polite column
    # train
    df_train['Formal'] = df_train['Consumer complaint narrative']\
                    .apply(lambda x: count_formal_informal(x) / float(len(x)))
    # test
    df_test['Formal'] = df_test['Consumer complaint narrative']\
                    .apply(lambda x: count_formal_informal(x) / float(len(x)))

    # calculate punctuation
    # train
    df_train['Punctuation'] = df_train['Consumer complaint narrative']\
                        .apply(lambda x: count_punctuation(x) / float(len(x)))
    # test
    df_test['Punctuation'] = df_test['Consumer complaint narrative']\
                        .apply(lambda x: count_punctuation(x) / float(len(x)))

    # count the number of times of first person occurances
    # train
    df_train['First person count'] = df_train['Consumer complaint narrative']\
                                    .apply(lambda x: count_I(x) / float(len(x)))
    # test
    df_test['First person count'] = df_test['Consumer complaint narrative']\
                                    .apply(lambda x: count_I(x) / float(len(x)))

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
    save_pickle(dict_train_test, '../data/feat_engineer.pkl')

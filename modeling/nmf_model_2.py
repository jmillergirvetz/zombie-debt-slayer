###Module to perform consumer narrative analyis####
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.stem import WordNetLemmatizer

from model_1 import load_pickle


def lemmer(doc):
    """
    Function that takes a a single doc and lemmatizes all words
    INPUT: ndarray of documents to be lemmatized
    OUPUT: List of lemmatized documents
    """
    wnl = WordNetLemmatizer()
    return ' '.join([wnl.lemmatize(word) for word in docs.split()])

def docs_lemmer(docs):
    """
    Function that takes numpy array of documents and applies lemmer function
    INPUT: numpy array of documents to be lemmatized
    OUPUT: List of lemmatized documents
    """
    L = []
    for i in xrange(len(docs)):
        L.append(lemmer(docs[i]))
    return L

def tfidf_vec(docs, stopwords='english', fit_transform=True):
    """
    Function that fits and transforms list of documents into a vecotrized \
    feature matrix
    INPUT: list of documents to be featurized
    OUTPUT: vectorized text feature matrix
    """
    tfidf = TfidfVectorizer(stop_words=stopwords)
    if fit_transform == True:
        X = tfidf.fit_transform(docs)
        return X, tfidf.vocabulary_
    elif fit_transform == False:
        X = tfidf.fit(docs)
        return X
    else:
        return None

def nmf_component_ids(mat, n_components=4):
    """
    Function to index the component matrix, H, of NMF
    INPUT: vectorized text feature matrix
    OUTPUT: indexed array of component importance of H
    """
    nmf = NMF(n_components=n_components)
    nmf.fit(mat)
    H = nmf.components_
    for arr in H:
        np.argsort(arr)[::-1]
    return H

def id_2_word(d):
    """
    Function that that reverses the keys and value of a dictionary
    INPUT: dictionary
    OUTPUT: dictionary with the keys and values reversed
    """
    tmp = {}
    for k in d:
        tmp[d[k]] = k
    return tmp
    # isn't this the same?
    # return [{v: k} for k, v in d.iteritems()][0]

def H_id_2_word(arr, d):
    """
    Function that creates topic dictionary
    INPUT: array and tfidf vocabulary dictionary
    OUTPUT: dictionary mapping topics to array indeces
    """
    tmp = {}
    for index in arr:
        tmp[arr[index]] = d[index]
    return tmp

def topic_model(H, d):
    """
    Function that creates a dictionary of topic dictionaries
    INPUT: NMF indexed component matrix, H, and tfidf vocab dict
    OUTPUT: dictionary of dictionaries mapping topics to array indeces
    """
    tmp = {}
    for arr, i in np.ndenumerate(H):
        tmp[i] = H_id_2_word(arr, d)
    return tmp


if __name__ == '__main__':

    # load dataframe with narratives
    df = load_pickle('../data/small_narr_mat_v1.pkl')

    # creats numpy array of narrative documents
    docs = df['Consumer complaint narrative'].values

    # creates list of lemmatized docuements
    lem_docs = docs_lemmer(docs)

    # creates vectorized text feature matrix and vocab list
    X, tfidf_vocab = tfidf_vec(lem_docs)

    # creates ndarray of NMF component matrix
    H_id_sort = nmf_component_ids(X, n_components=4)

    # creates vocab dict with the vectorized text feature matrix and indeces
    d = id_2_word(tfidf_vocab)

    # computes topic modeling by matching vocab to indexed component matrix, H
    d_topics = topic_model(H_id_sort, d)

    ### d_topics could then be used to create vocabulary lists for each
    # original document which could help improve topic modeling for the process

import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
#from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer



if __name__ == "__main__":

    col_data_types = {'Consumer complaint narrative':np.str_, \
                      'Company public response':np.str_, \
                      'Consumer consent provided?':np.str_}

    df = pd.read_csv('../data/Consumer_Complaints.csv', dtype=col_data_types)

    docs = df['Company'].unique()
    zips = df['ZIP code']

    n_clusters = 100

    #count vectorizer

    vectorizer = TfidfVectorizer()
    document_term_mat = vectorizer.fit_transform(docs)
    feature_words = vectorizer.get_feature_names()

    nmf = NMF(n_components=n_topics)
    W = nmf.fit_transform(document_term_mat)
    H = nmf.components_

    km = KMeans(n_clusters=n_clusters, n_init=3, init='k-means++')
    km.fit_transform(document_term_mat)
    print set(km.labels_)

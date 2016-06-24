import pandas as pd
import numpy as np

# Ovesampling module found at | https://github.com/fmfn/UnbalancedDataset
from unbalanced_dataset import over_sampling

"""
Script to oversample underrepresented classes. \
The random_over_sample function below only oversamples between binary classes. \
Thus, for multi-classes classification problems the resampled outputs will need \
to be run through n - 1 times.
"""


def random_over_sample(X, y):
    """"
    Function that oversamples a single underrepresented class
    INPUT: X feature matrix
    OUTPUT: y binary labels
    """"
    ros = over_sampling.random_over_sampler.RandomOverSampler()
    ros.fit(X, y)
    X_resampled, y_resampled = ros.transform(X, y)

    return X_resampled, y_resampled


if __name__ == '__main__':
    

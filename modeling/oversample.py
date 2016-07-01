###Module to oversample: https://github.com/fmfn/UnbalancedDataset###
import pandas as pd
import numpy as np

from unbalanced_dataset import over_sampling


def random_over_sample(X, y):
    """
    Function that proportionately oversamples all underrepresented classes
    INPUT: X feature matrix
    OUTPUT: X and y resampled feature matrix and output labels respectively
    """
    ros = over_sampling.RandomOverSampler()
    ros.fit(X, y)
    X_resampled, y_resampled = ros.transform(X, y)

    return X_resampled, y_resampled


if __name__ == '__main__':
    main()

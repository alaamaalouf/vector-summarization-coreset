import pandas as pd
import numpy as np
import os


REPS = 1
NUM_SAMPLES = 10


def readDataset(file_name):
    file_path = r'datasets\\' + file_name
    if '.csv' in file_name:
        P = pd.read_csv(file_path).values

    return P


def getNormalizedWeightedSet(P, weights):
    new_weights = weights / np.linalg.norm(weights, ord=1)
    mu = np.sum(np.multiply(P, new_weights[:, np.newaxis]), 0)
    sigma = np.sqrt(np.sum(np.multiply(np.sum((P - mu) ** 2, 1), new_weights)))

    return (P - mu) / sigma, new_weights, mu, sigma


def generateSampleSizes(n):
    min_size = int(np.log(n) ** 2)
    max_size = n//10

    pass


def createDirectory(directory_name):
    """
    ##################### createDirectory ####################
    Input:
        - path: A string containing a path of an directory to be created at.

    Output:
        - None

    Description:
        This process is responsible creating an empty directory at a given path.
    """
    full_path = r'results\\'+directory_name
    try:
        os.makedirs(full_path)
    except OSError:
        if not os.path.isdir(full_path):
            raise


if __name__ == '__main__':
    P = np.random.rand(100, 2)
    weights = np.ones((P.shape[0], ))
    Q, W, _, _ = getNormalizedWeightedSet(P, weights)
    assert(np.sum(W) == 1, 'Sum of weights must be 1!')
    assert(np.linalg.norm(np.sum(np.multiply(Q, W[:, np.newaxis]), 0)) == 0, 'The weighted sum must be the origin!')
    assert(np.sum(np.multiply(W, np.sum(Q ** 2, 1))) == 1, 'Variance of points must be 1!')




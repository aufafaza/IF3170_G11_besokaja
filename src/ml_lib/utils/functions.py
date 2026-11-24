import numpy as np


class Sigmoid:
    def __call__(self, x):
        # clipping prevents exp(-x) from overflowing for large negative x
        return 1 / (1 + np.exp(-np.clip(x, -700, 700)))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


def minkowski_distance(A, B, p):
    '''
    p is the degree of distance used, i.e.
    p = 1:  manhattan distance
    p = 2:euclidean distance
    p = inf: chebyshev distance
    Returns the minkowski distance of A and B
    '''
    difference = np.abs(np.asarray(A, dtype=float) - np.asarray(B, dtype=float))
    if p == np.inf:
        return np.max(difference)
    return np.power(np.sum(difference ** p), 1 / p)

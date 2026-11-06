import numpy as np

from ml_lib.model.base import BaseModel
from ml_lib.utils.operation import minkowski_distance


class KNN(BaseModel):
    """
    k: int 
        assume the default hyperparameter k = 5 
    p: int 
        the chosen minkowski distance p parameter 
            p = 1: manhattan distance 
            p = 2: euclidean distance 
            p = inf: chebysev distance 
    """

    def __init__(self, k: int, p: int):
        self.k = k
        self.p = p
        self._X_train = None
        self._y_train = None

    def fit(self, X, y) -> BaseModel:
        self._X_train = np.asarray(X)
        self._y_train = np.asarray(y)
        return self

    def _vote(self, neighbors):
        """most common class among its neighbors"""
        counts = np.bincount(neighbors.astype("int"))
        return counts.argmax()

    def predict(self, X) -> np.ndarray:
        """Predict labels for the samples in X."""
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("KNN must be fitted before calling predict")

        X = np.asarray(X)
        y_pred = np.empty(X.shape[0])

        for i, sample in enumerate(X):
            idx = np.argsort(
                [minkowski_distance(sample, x, self.p) for x in self._X_train]
            )[:self.k]
            neighbors = self._y_train[idx]
            y_pred[i] = self._vote(neighbors)

        return y_pred

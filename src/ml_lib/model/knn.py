import numpy as np

from ml_lib.model.base import BaseModel


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
        labels, counts = np.unique(neighbors, return_counts=True)
        return labels[counts.argmax()]

    def neighbor_labels(self, X):
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("KNN must be fitted before calling predict")

        if self.k <= 0:
            raise ValueError("k must be > 0")

        count = min(self.k, len(self._y_train))
        for sample in np.asarray(X):
            distances = np.power(
                np.sum(np.abs(self._X_train - sample) ** self.p, axis=1),
                1 / self.p,
            )
            indices = np.argpartition(distances, count - 1)[:count]
            yield self._y_train[indices]

    def predict(self, X) -> np.ndarray:
        return np.asarray(
            [self._vote(labels) for labels in self.neighbor_labels(X)],
            dtype=self._y_train.dtype,
        )

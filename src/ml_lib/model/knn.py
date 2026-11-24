from numbers import Integral

import numpy as np

from ml_lib.model.base import BaseModel

_BATCH_SIZE = 512


class KNN(BaseModel):
    """
    k-nearest neighbors classifier using Minkowski distance.

    k: int
        number of neighbors, default k = 5
    p: int, float, or np.inf
        the chosen minkowski distance p parameter, default p = 2
            p = 1: manhattan distance
            p = 2: euclidean distance
            p = inf: chebysev distance

    Equal vote counts resolve to the smallest class label.
    """

    def __init__(self, k: int = 5, p: float = 2):
        self.k = k
        self.p = p
        self._X_train = None
        self._y_train = None

    def fit(self, X, y) -> BaseModel:
        self._validate_hyperparameters()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.size:
            raise ValueError(f"X and y have inconsistent lengths: {X.shape[0]} != {y.size}")
        if y.size == 0:
            raise ValueError("Cannot fit on an empty dataset")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity. Please impute missing values first")

        self._X_train = X
        self._y_train = y
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self._y_encoded = encoded.ravel()
        # centering keeps the squared-distance expansion accurate for large offsets
        self._center = X.mean(axis=0)
        self._X_centered = X - self._center
        self._train_squared_norms = np.einsum("ij,ij->i", self._X_centered, self._X_centered)
        return self

    def _validate_hyperparameters(self):
        if not isinstance(self.k, Integral) or self.k <= 0:
            raise ValueError("k must be a positive integer")
        if not self.p >= 1:
            raise ValueError("p must be at least 1")

    def _prepare_queries(self, X):
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("KNN must be fitted before calling predict")
        self._validate_hyperparameters()

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self._X_train.shape[1]:
            raise ValueError(
                f"X must have shape (n_samples, {self._X_train.shape[1]}), got {X.shape}"
            )
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity. Please impute missing values first")
        return X - self._center

    def _rank_distances(self, queries):
        """Distances without the final root, which preserves neighbor order."""
        train = self._X_centered
        if self.p == 2:
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
            squared = (
                (queries * queries).sum(axis=1, keepdims=True)
                + self._train_squared_norms
                - 2 * queries @ train.T
            )
            return np.maximum(squared, 0)

        distances = np.zeros((queries.shape[0], train.shape[0]))
        for feature in range(train.shape[1]):
            difference = np.abs(queries[:, feature, None] - train[:, feature])
            if self.p == np.inf:
                np.maximum(distances, difference, out=distances)
            elif self.p == 1:
                distances += difference
            else:
                distances += difference ** self.p
        return distances

    def _neighbor_index_batches(self, queries):
        count = min(int(self.k), self._y_train.size)
        for start in range(0, queries.shape[0], _BATCH_SIZE):
            distances = self._rank_distances(queries[start:start + _BATCH_SIZE])
            yield np.argpartition(distances, count - 1, axis=1)[:, :count]

    def _vote(self, neighbor_indices):
        """most common class among each row of neighbors"""
        labels = self._y_encoded[neighbor_indices]
        n_rows, n_neighbors = labels.shape
        n_classes = self.classes_.size
        rows = np.repeat(np.arange(n_rows), n_neighbors)
        counts = np.bincount(
            rows * n_classes + labels.ravel(), minlength=n_rows * n_classes
        ).reshape(n_rows, n_classes)
        return counts.argmax(axis=1)

    def neighbor_labels(self, X):
        queries = self._prepare_queries(X)
        for indices in self._neighbor_index_batches(queries):
            yield from self._y_train[indices]

    def predict(self, X) -> np.ndarray:
        queries = self._prepare_queries(X)
        predictions = np.empty(queries.shape[0], dtype=np.int64)
        start = 0
        for indices in self._neighbor_index_batches(queries):
            predictions[start:start + indices.shape[0]] = self._vote(indices)
            start += indices.shape[0]
        return self.classes_[predictions]

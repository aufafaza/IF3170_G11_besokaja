from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin

from ml_lib.model.base import BaseModel
from ml_lib.utils.tabular import TabularEncoder, as_frame, resolve_feature_mask


def _logsumexp(a: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable log(sum(exp(a))) using the max-shift trick."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


class GaussianNB(ClassifierMixin, BaseModel):
    """Gaussian Naive Bayes"""

    def __init__(self, priors=None, var_smoothing=1e-9):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y) -> GaussianNB:
        X, y = self._validate_fit_input(X, y)
        y_idx = self._fit_priors(y)
        self._fit_gaussian(X, y_idx)
        return self

    def _validate_fit_input(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {X.shape[0]} != {y.shape[0]}"
            )
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on an empty dataset")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity. Please impute missing values first")

        self.n_features_in_ = X.shape[1]
        return X, y

    def _fit_priors(self, y) -> np.ndarray:
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        y_idx = y_idx.ravel()
        n_classes = len(self.classes_)
        self.class_count_ = np.bincount(y_idx, minlength=n_classes).astype(float)

        if self.priors is None:
            self.class_prior_ = self.class_count_ / self.class_count_.sum()
        else:
            priors = np.asarray(self.priors, dtype=float)
            if priors.shape != (n_classes,):
                raise ValueError(
                    f"priors must have one entry per class ({n_classes}), "
                    f"got shape {priors.shape}"
                )
            if (priors < 0).any():
                raise ValueError("priors must be non-negative")
            if not np.isclose(priors.sum(), 1.0):
                raise ValueError("The sum of the priors should be 1")
            self.class_prior_ = priors
        return y_idx

    def _fit_gaussian(self, Xc: np.ndarray, y_idx: np.ndarray) -> None:
        """Estimate per-class mean and variance of every column in Xc."""
        n_classes = len(self.classes_)
        n_features = Xc.shape[1]

        self.epsilon_ = (
            self.var_smoothing * Xc.var(axis=0).max() if n_features else 0.0
        )
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        for c in range(n_classes):
            X_c = Xc[y_idx == c]
            self.theta_[c] = X_c.mean(axis=0)
            self.var_[c] = X_c.var(axis=0)
        self.var_ += self.epsilon_

        # Guard against division by zero when every feature is constant
        self.var_[self.var_ == 0] = np.finfo(float).eps

    def _log_prior(self) -> np.ndarray:
        with np.errstate(divide="ignore"):
            return np.log(self.class_prior_)

    def _gaussian_jll(self, Xc: np.ndarray) -> np.ndarray:
        n_classes = len(self.classes_)
        jll = np.zeros((Xc.shape[0], n_classes))
        if Xc.shape[1] == 0:
            return jll
        for c in range(n_classes):
            log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[c]))
            sq = np.sum((Xc - self.theta_[c]) ** 2 / self.var_[c], axis=1)
            jll[:, c] = log_norm - 0.5 * sq
        return jll

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """log P(c) + sum_j log P(x_j | c), shape (n_samples, n_classes)."""
        return self._log_prior() + self._gaussian_jll(X)

    def _validate_predict_input(self, X) -> np.ndarray:
        if not hasattr(self, "classes_"):
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before calling predict"
            )
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} "
                f"was fitted with {self.n_features_in_} features"
            )
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity. Please impute missing values first")
        return X

    def predict_log_proba(self, X) -> np.ndarray:
        """Log posterior log P(c | x), columns ordered as classes_."""
        jll = self._joint_log_likelihood(self._validate_predict_input(X))
        return jll - _logsumexp(jll, axis=1)[:, None]

    def predict_proba(self, X) -> np.ndarray:
        """Posterior P(c | x), columns ordered as classes_."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        """Maximum a posteriori class for each sample."""
        jll = self._joint_log_likelihood(self._validate_predict_input(X))
        return self.classes_[np.argmax(jll, axis=1)]

class DiscreteNB(GaussianNB):
    """
    Frequency-based Naive Bayes for categorical attributes.

    Numeric columns are discretized during fit:
        - integer columns with at most max_categories values keep those values
        - other numeric columns are split into n_bins bins

    Probabilities and classification:
        P(c) = count(c) / N
        P(v | c) = (count(v, c) + alpha) / (count(c) + alpha * |V|)
        prediction = argmax_c [log P(c) + sum_j log P(x_j | c)]

    alpha=1 is +1 (Laplace) smoothing. Nulls are their own attribute value.
    Values unseen during fit use count(v, c) = 0.
    """

    def __init__(
        self,
        alpha=1.0,
        n_bins=10,
        strategy="quantile",
        max_categories=10,
        categorical_features=None,
        priors=None,
    ):
        self.alpha = alpha
        self.n_bins = n_bins
        self.strategy = strategy
        self.max_categories = max_categories
        self.categorical_features = categorical_features
        self.priors = priors

    def fit(self, X, y) -> DiscreteNB:
        self._validate_hyperparameters()
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

        self._encoder = TabularEncoder(self.categorical_features)
        encoded = self._encoder.fit_transform(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if encoded.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y have inconsistent lengths: {encoded.shape[0]} != {y.shape[0]}"
            )
        if encoded.shape[0] == 0:
            raise ValueError("Cannot fit on an empty dataset")

        self.n_features_in_ = encoded.shape[1]
        null_mask = as_frame(X).isna().to_numpy()
        y_idx = self._fit_priors(y)
        self._fit_value_maps(encoded, null_mask)
        self._fit_frequency_tables(self._value_codes(encoded, null_mask), y_idx)
        return self

    def _validate_hyperparameters(self):
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")
        if self.n_bins < 2 or int(self.n_bins) != self.n_bins:
            raise ValueError("n_bins must be an integer of at least 2")
        if self.strategy not in ("quantile", "uniform"):
            raise ValueError("strategy must be 'quantile' or 'uniform'")
        if self.max_categories < 1:
            raise ValueError("max_categories must be at least 1")

    def _fit_value_maps(self, encoded, null_mask):
        """Learn how each column maps to discrete attribute values."""
        n_features = encoded.shape[1]
        self.feature_kinds_ = []
        self.bin_edges_ = [None] * n_features
        self.discrete_values_ = [None] * n_features
        self.n_values_ = np.zeros(n_features, dtype=np.int64)
        self.missing_codes_ = np.full(n_features, -1, dtype=np.int64)

        for j in range(n_features):
            if self._encoder.categorical_mask_[j]:
                kind = "categorical"
                n_observed = len(self._encoder.categories_[j])
            else:
                column = encoded[:, j]
                observed = column[~np.isnan(column)]
                if np.isinf(observed).any():
                    raise ValueError("X contains infinity")
                values = np.unique(observed)
                if values.size <= self.max_categories and np.all(values == np.round(values)):
                    kind = "discrete"
                    self.discrete_values_[j] = values
                    n_observed = values.size
                else:
                    kind = "binned"
                    self.bin_edges_[j] = self._bin_edges(observed)
                    n_observed = self.bin_edges_[j].size + 1

            self.feature_kinds_.append(kind)
            self.n_values_[j] = n_observed
            if null_mask[:, j].any():
                self.missing_codes_[j] = n_observed
                self.n_values_[j] += 1

    def _bin_edges(self, observed):
        inner = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        if self.strategy == "quantile":
            edges = np.unique(np.quantile(observed, inner))
        else:
            edges = observed.min() + inner * np.ptp(observed)

        # merge empty bins so |V| counts only real attribute values
        counts = np.bincount(
            np.searchsorted(edges, observed, side="right"), minlength=edges.size + 1
        )
        return edges[counts[:-1] > 0]

    def _value_codes(self, encoded, null_mask):
        """Attribute-value codes per cell. Code -1 means unseen during fit."""
        codes = np.empty(encoded.shape, dtype=np.int64)
        for j, kind in enumerate(self.feature_kinds_):
            column = encoded[:, j]
            observed = ~np.isnan(column)
            values = np.full(column.shape, -1, dtype=np.int64)
            if kind == "binned":
                values[observed] = np.searchsorted(
                    self.bin_edges_[j], column[observed], side="right"
                )
            elif kind == "discrete":
                known = self.discrete_values_[j]
                if known.size:
                    position = np.minimum(np.searchsorted(known, column[observed]), known.size - 1)
                    values[observed] = np.where(known[position] == column[observed], position, -1)
            else:
                values[observed] = column[observed].astype(np.int64)
            values[null_mask[:, j]] = self.missing_codes_[j]
            codes[:, j] = values
        return codes

    def _fit_frequency_tables(self, codes, y_idx):
        alpha = max(self.alpha, 1e-10)
        n_classes = len(self.classes_)
        self.category_count_ = []
        self.feature_log_prob_ = []
        self._log_prob_lookup = []

        for j, n_values in enumerate(self.n_values_):
            counts = np.bincount(
                y_idx * n_values + codes[:, j], minlength=n_classes * n_values
            ).reshape(n_classes, n_values).astype(float)
            denominator = self.class_count_[:, None] + alpha * n_values
            log_prob = np.log(counts + alpha) - np.log(denominator)
            unseen_log_prob = np.log(alpha) - np.log(denominator)
            self.category_count_.append(counts)
            self.feature_log_prob_.append(log_prob)
            # the final column is reached by code -1
            self._log_prob_lookup.append(np.hstack((log_prob, unseen_log_prob)))

    def _validate_predict_input(self, X) -> np.ndarray:
        if not hasattr(self, "classes_"):
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before calling predict"
            )
        encoded = self._encoder.transform(X)
        return self._value_codes(encoded, as_frame(X).isna().to_numpy())

    def _joint_log_likelihood(self, codes: np.ndarray) -> np.ndarray:
        jll = np.zeros((codes.shape[0], len(self.classes_))) + self._log_prior()
        for j, lookup in enumerate(self._log_prob_lookup):
            jll += lookup[:, codes[:, j]].T
        return jll

class MixedNB(GaussianNB):
    """
    Naive Bayes with Gaussian likelihoods for continuous features and
    Laplace-smoothed categorical likelihoods for discrete features.
    """

    def __init__(
        self,
        categorical_features=None,
        alpha=1.0,
        max_categories=10,
        priors=None,
        var_smoothing=1e-9,
    ):
        self.categorical_features = categorical_features
        self.alpha = alpha
        self.max_categories = max_categories
        self.priors = priors
        self.var_smoothing = var_smoothing

    def fit(self, X, y) -> MixedNB:
        if self.alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {self.alpha}")

        X, y = self._validate_fit_input(X, y)
        cat_mask = self._resolve_categorical_mask(X)
        self.categorical_features_ = np.flatnonzero(cat_mask)
        self.continuous_features_ = np.flatnonzero(~cat_mask)

        y_idx = self._fit_priors(y)
        self._fit_gaussian(X[:, self.continuous_features_], y_idx)
        self._fit_categorical(X[:, self.categorical_features_], y_idx)
        return self

    def _resolve_categorical_mask(self, X: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        spec = self.categorical_features

        if spec is None:
            return np.array(
                [self._looks_discrete(X[:, j]) for j in range(n_features)],
                dtype=bool,
            )
        return resolve_feature_mask(spec, n_features, getattr(self, "feature_names_in_", None))

    def _looks_discrete(self, col: np.ndarray) -> bool:
        return bool(
            np.all(col == np.round(col))
            and np.unique(col).size <= self.max_categories
        )

    def _fit_categorical(self, Xd: np.ndarray, y_idx: np.ndarray) -> None:
        """Build the smoothed frequency table of every discrete column."""
        alpha = max(self.alpha, 1e-10)
        n_classes = len(self.classes_)
        self.categories_ = []
        self.category_count_ = []
        self.feature_log_prob_ = []
        self.unseen_log_prob_ = []

        for j in range(Xd.shape[1]):
            cats, codes = np.unique(Xd[:, j], return_inverse=True)
            counts = np.bincount(
                y_idx * cats.size + codes.ravel(), minlength=n_classes * cats.size
            ).reshape(n_classes, cats.size).astype(float)
            denominator = self.class_count_[:, None] + alpha * cats.size
            self.categories_.append(cats)
            self.category_count_.append(counts)
            self.feature_log_prob_.append(np.log(counts + alpha) - np.log(denominator))
            self.unseen_log_prob_.append(np.log(alpha) - np.log(denominator).ravel())

    def _categorical_jll(self, Xd: np.ndarray) -> np.ndarray:
        """sum_j log P(x_j | c) over the discrete columns."""
        jll = np.zeros((Xd.shape[0], len(self.classes_)))
        tables = zip(self.categories_, self.feature_log_prob_, self.unseen_log_prob_)
        for j, (cats, log_prob, unseen_log_prob) in enumerate(tables):
            col = Xd[:, j]
            pos = np.minimum(np.searchsorted(cats, col), cats.size - 1)
            seen = cats[pos] == col
            jll[seen] += log_prob[:, pos[seen]].T
            jll[~seen] += unseen_log_prob  # count 0 in the smoothed frequency
        return jll

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        return (
            self._log_prior()
            + self._gaussian_jll(X[:, self.continuous_features_])
            + self._categorical_jll(X[:, self.categorical_features_])
        )


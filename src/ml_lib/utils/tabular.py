from __future__ import annotations

import numpy as np
import pandas as pd

_NUMERIC_INFERRED_TYPES = {
    "boolean",
    "decimal",
    "empty",
    "floating",
    "integer",
    "mixed-integer-float",
}


def as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    array = X if isinstance(X, np.ndarray) else np.asarray(X, dtype=object)
    if array.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {array.shape}")
    return pd.DataFrame(array)


def is_categorical_column(column: pd.Series) -> bool:
    dtype = column.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return True
    if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        return False
    return pd.api.types.infer_dtype(column, skipna=True) not in _NUMERIC_INFERRED_TYPES


def resolve_feature_mask(spec, n_features: int, feature_names=None) -> np.ndarray:
    """Turn indices, column names, or a boolean mask into a boolean mask."""
    spec = np.asarray(spec)
    mask = np.zeros(n_features, dtype=bool)
    if spec.size == 0:
        return mask
    if spec.ndim != 1:
        raise ValueError("categorical_features must be one-dimensional")

    if spec.dtype == bool:
        if spec.shape != (n_features,):
            raise ValueError(
                f"Boolean categorical_features must have length {n_features}, "
                f"got {spec.shape[0]}"
            )
        return spec.copy()

    if spec.dtype.kind in "iu":
        if ((spec < -n_features) | (spec >= n_features)).any():
            raise ValueError(
                f"categorical_features indices out of range for {n_features} features"
            )
        mask[spec] = True
        return mask

    if spec.dtype.kind in "UO":
        if feature_names is None:
            raise ValueError(
                "categorical_features given as column names requires X to be a DataFrame"
            )
        lookup = {name: i for i, name in enumerate(feature_names)}
        missing = [name for name in spec if name not in lookup]
        if missing:
            raise ValueError(f"Unknown columns in categorical_features: {missing}")
        mask[[lookup[name] for name in spec]] = True
        return mask

    raise ValueError(
        "categorical_features must be None, a list of indices, "
        "a list of column names, or a boolean mask"
    )


class TabularEncoder:
    """
    Encode numeric columns as floats and categorical columns as integer codes.

    Missing values and categories unseen during fitting become NaN.
    """

    def __init__(self, categorical_features=None):
        self.categorical_features = categorical_features

    def fit_transform(self, X) -> np.ndarray:
        if self._is_plain_numeric(X) and self.categorical_features is None:
            if X.ndim != 2:
                raise ValueError(f"X must be 2D, got shape {X.shape}")
            self.n_features_in_ = X.shape[1]
            self.categorical_mask_ = np.zeros(X.shape[1], dtype=bool)
            self.categories_ = [None] * X.shape[1]
            return X.astype(float)

        frame = as_frame(X)
        n_features = frame.shape[1]
        names = frame.columns if isinstance(X, pd.DataFrame) else None
        requested = (
            np.zeros(n_features, dtype=bool)
            if self.categorical_features is None
            else resolve_feature_mask(self.categorical_features, n_features, names)
        )

        self.n_features_in_ = n_features
        self.categorical_mask_ = requested.copy()
        self.categories_ = [None] * n_features
        encoded = np.empty(frame.shape, dtype=float)
        for j in range(n_features):
            column = frame.iloc[:, j]
            if requested[j] or is_categorical_column(column):
                codes, categories = pd.factorize(column.astype(object), use_na_sentinel=True)
                self.categorical_mask_[j] = True
                self.categories_[j] = pd.Index(categories, dtype=object)
                encoded[:, j] = np.where(codes < 0, np.nan, codes)
            else:
                encoded[:, j] = column.to_numpy(dtype=float, na_value=np.nan)
        return encoded

    def transform(self, X) -> np.ndarray:
        if self._is_plain_numeric(X) and not self.categorical_mask_.any():
            if X.ndim != 2 or X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X must have shape (n_samples, {self.n_features_in_}), got {X.shape}"
                )
            return X.astype(float)

        frame = as_frame(X)
        if frame.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {frame.shape[1]} features, expected {self.n_features_in_}"
            )

        encoded = np.empty(frame.shape, dtype=float)
        for j, categories in enumerate(self.categories_):
            column = frame.iloc[:, j]
            if categories is None:
                encoded[:, j] = column.to_numpy(dtype=float, na_value=np.nan)
            else:
                codes = categories.get_indexer(column.astype(object))
                encoded[:, j] = np.where(codes < 0, np.nan, codes)
        return encoded

    @staticmethod
    def _is_plain_numeric(X) -> bool:
        return isinstance(X, np.ndarray) and X.dtype.kind in "biuf"

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import RegressorTags, check_random_state

from ml_lib.model.base import BaseModel
from ml_lib.utils.tabular import TabularEncoder


class TreeNode:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None,
        *, value=None, n_samples=0, impurity=0.0, leaf_value=None,
        missing_left=False, left_categories=None,
        ):
        self.feature=feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.impurity = impurity
        self.leaf_value = leaf_value
        self.missing_left = missing_left
        self.left_categories = left_categories

    def is_leaf_node(self):
        return self.value is not None


class _Split(NamedTuple):
    score: float
    feature: int
    threshold: float | None
    missing_left: bool
    left_categories: np.ndarray | None


class DecisionTree(ClassifierMixin, BaseModel):
    """
    ID3-style binary decision tree using information gain.

    Numeric, categorical, and missing values are handled during fit and predict.
    Each split learns which child receives missing values. Categories unseen
    during fit follow that same missing-value direction.
    """

    def __init__(
        self, min_samples_split=2, max_depth=100, n_features=None,
        categorical_features=None, random_state=None,
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.categorical_features = categorical_features
        self.random_state = random_state
        self._n_features = None
        self.root=None

    def fit(self, X, y) -> DecisionTree:
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

        self._encoder = TabularEncoder(self.categorical_features)
        X = np.asfortranarray(self._encoder.fit_transform(X))
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if X.shape[0] != y.size:
            raise ValueError(f"X and y have inconsistent lengths: {X.shape[0]} != {y.size}")
        if y.size == 0:
            raise ValueError("Cannot fit on an empty dataset")

        self._n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.n_features_in_ = X.shape[1]
        stats = self._target_stats(y)
        rng = check_random_state(self.random_state)

        self.root = self._grow_tree(X, stats, np.arange(y.size), rng)
        self._compile_tree()
        return self

    def _target_stats(self, y):
        """One-hot class counts, so cumulative sums give child class counts."""
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self._value_dtype = self.classes_.dtype
        return np.eye(self.classes_.size)[encoded.ravel()]

    def _grow_tree(self, X, stats, indices, rng, depth=0):
        n_samples = indices.size
        node_stats = stats[indices]
        totals = node_stats.sum(axis=0)
        impurity = float(self._impurity(totals))
        leaf_value = self._leaf_value(totals)

        # check stopping criteria
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or self._is_pure(node_stats, totals)
        ):
            return TreeNode(value=leaf_value, n_samples=n_samples, impurity=impurity, leaf_value=leaf_value)

        # find the best split (gain)
        feat_indices = rng.choice(X.shape[1], self._n_features, replace=False)

        split = self._best_split(X, node_stats, indices, feat_indices, totals, impurity)
        if split is None:
            return TreeNode(value=leaf_value, n_samples=n_samples, impurity=impurity, leaf_value=leaf_value)

        # create child nodes
        goes_left = self._goes_left(X[indices, split.feature], split)
        left = self._grow_tree(X, stats, indices[goes_left], rng, depth + 1)
        right = self._grow_tree(X, stats, indices[~goes_left], rng, depth + 1)

        return TreeNode(
            split.feature, split.threshold, left, right,
            n_samples=n_samples, leaf_value=leaf_value, impurity=impurity,
            missing_left=split.missing_left, left_categories=split.left_categories,
        )

    def _is_pure(self, node_stats, totals):
        return np.count_nonzero(totals) == 1

    def _sample_counts(self, stats):
        return stats.sum(axis=-1)

    def _impurity(self, stats):
        # entropy: -sum(p(X).log(p(X)))
        p = stats / stats.sum(axis=-1, keepdims=True)
        safe_p = np.where(p > 0, p, 1.0)
        return -(p * np.log(safe_p)).sum(axis=-1)

    def _leaf_value(self, totals):
        return self.classes_[np.argmax(totals)]

    def _split_score(self, gain, n_left, n_right, n_total):
        return gain

    def _thresholds(self, lower, upper):
        return lower

    def _category_rank(self, category_stats):
        """Sort key giving optimal prefix splits, or None if no such order exists."""
        if category_stats.shape[1] != 2:
            return None
        return category_stats[:, 1] / category_stats.sum(axis=1)

    def _score_candidates(self, left, totals, parent_impurity):
        right = totals - left
        n_total = self._sample_counts(totals)
        n_left = self._sample_counts(left)
        n_right = n_total - n_left
        child_impurity = (
            n_left * self._impurity(left) + n_right * self._impurity(right)
        ) / n_total
        return self._split_score(parent_impurity - child_impurity, n_left, n_right, n_total)

    def _best_split(self, X, node_stats, indices, feat_indices, totals, parent_impurity):
        best = None
        for feature in feat_indices:
            column = X[indices, feature]
            missing = np.isnan(column)
            has_missing = bool(missing.any())
            values = column[~missing] if has_missing else column
            value_stats = node_stats[~missing] if has_missing else node_stats
            if values.size == 0:
                continue

            if self._encoder.categorical_mask_[feature]:
                left, order, is_prefix = self._categorical_candidates(values, value_stats, has_missing)
                thresholds = None
            else:
                left, thresholds = self._numeric_candidates(values, value_stats, has_missing)
                order, is_prefix = None, False
            if left.shape[0] == 0:
                continue

            missing_stats = node_stats[missing].sum(axis=0) if has_missing else None
            score, index, missing_left = self._select_candidate(
                left, missing_stats, totals, parent_impurity
            )
            if best is None or score > best.score:
                threshold = None if thresholds is None else float(thresholds[index])
                categories = (
                    None if order is None else self._left_categories(order, index, is_prefix)
                )
                best = _Split(float(score), int(feature), threshold, missing_left, categories)
        return best

    def _numeric_candidates(self, values, stats, has_missing):
        """Left-child stats and thresholds at every boundary between sorted values."""
        order = np.argsort(values)
        ordered = values[order]
        cumulative = np.cumsum(stats[order], axis=0)
        boundaries = np.flatnonzero(ordered[1:] != ordered[:-1])
        left = cumulative[boundaries]
        thresholds = self._thresholds(ordered[boundaries], ordered[boundaries + 1])
        if has_missing:
            # final candidate: observed values on the left, missing values on the right
            left = np.vstack((left, cumulative[-1]))
            thresholds = np.append(thresholds, np.inf)
        return left, thresholds

    def _categorical_candidates(self, codes, stats, has_missing):
        present, inverse = np.unique(codes.astype(np.int64), return_inverse=True)
        category_stats = np.column_stack([
            np.bincount(inverse.ravel(), weights=stats[:, column], minlength=present.size)
            for column in range(stats.shape[1])
        ])

        rank = self._category_rank(category_stats)
        if rank is not None:
            sort_order = np.argsort(rank, kind="stable")
            order = present[sort_order]
            cumulative = np.cumsum(category_stats[sort_order], axis=0)
            left = cumulative[:-1]
            is_prefix = True
            all_observed = cumulative[-1]
        else:
            # one category against the rest for multiclass targets
            order = present
            left = category_stats if present.size > 1 else category_stats[:0]
            is_prefix = False
            all_observed = category_stats.sum(axis=0)

        if has_missing:
            left = np.vstack((left, all_observed))
        return left, order, is_prefix

    @staticmethod
    def _left_categories(order, index, is_prefix):
        if is_prefix:
            return order[:index + 1]
        if index >= order.size:
            return order
        return order[[index]]

    def _select_candidate(self, left, missing_stats, totals, parent_impurity):
        scores = self._score_candidates(left, totals, parent_impurity)
        index = int(np.argmax(scores))
        score = scores[index]

        if missing_stats is None:
            # no training nulls at this node: unseen nulls follow the larger child
            n_left = self._sample_counts(left[index])
            return score, index, bool(n_left >= self._sample_counts(totals) - n_left)

        missing_left = False
        if left.shape[0] > 1:
            # the last candidate already separates observed from missing values
            left_with_missing = left[:-1] + missing_stats
            alternative = self._score_candidates(left_with_missing, totals, parent_impurity)
            alternative_index = int(np.argmax(alternative))
            if alternative[alternative_index] > score:
                score = alternative[alternative_index]
                index = alternative_index
                missing_left = True
        return score, index, missing_left

    @staticmethod
    def _goes_left(column, split):
        if split.left_categories is not None:
            left = np.isin(column, split.left_categories)
        else:
            left = column <= split.threshold
        return left | (np.isnan(column) & split.missing_left)

    def _compile_tree(self):
        """Store the fitted tree in flat arrays for batched prediction."""
        nodes = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            nodes.append(node)
            if not node.is_leaf_node():
                stack.extend((node.right, node.left))

        positions = {id(node): i for i, node in enumerate(nodes)}
        n_nodes = len(nodes)
        feature = np.full(n_nodes, -1, dtype=np.int64)
        threshold = np.full(n_nodes, np.nan)
        left = np.full(n_nodes, -1, dtype=np.int64)
        right = np.full(n_nodes, -1, dtype=np.int64)
        missing_left = np.zeros(n_nodes, dtype=bool)
        category_row = np.full(n_nodes, -1, dtype=np.int64)
        values = np.empty(n_nodes, dtype=self._value_dtype)

        category_nodes = [node for node in nodes if node.left_categories is not None]
        width = max(
            (len(c) for c in self._encoder.categories_ if c is not None), default=1
        )
        category_table = np.zeros((len(category_nodes), width), dtype=bool)
        row = 0
        for i, node in enumerate(nodes):
            values[i] = node.leaf_value
            if node.is_leaf_node():
                continue
            feature[i] = node.feature
            left[i] = positions[id(node.left)]
            right[i] = positions[id(node.right)]
            missing_left[i] = node.missing_left
            if node.left_categories is None:
                threshold[i] = node.threshold
            else:
                category_row[i] = row
                category_table[row, node.left_categories] = True
                row += 1

        self.tree_ = {
            "feature": feature,
            "threshold": threshold,
            "left": left,
            "right": right,
            "missing_left": missing_left,
            "category_row": category_row,
            "category_table": category_table,
            "value": values,
        }

    def predict(self, X) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("DecisionTree must be fitted before calling predict")

        X = self._encoder.transform(X)
        tree = self.tree_
        feature = tree["feature"]
        category_row = tree["category_row"]
        node = np.zeros(X.shape[0], dtype=np.int64)
        active = np.flatnonzero(feature[node] >= 0)

        while active.size:
            current = node[active]
            values = X[active, feature[current]]
            missing = np.isnan(values)
            rows = category_row[current]

            goes_left = values <= tree["threshold"][current]
            categorical = np.flatnonzero((rows >= 0) & ~missing)
            if categorical.size:
                goes_left[categorical] = tree["category_table"][
                    rows[categorical], values[categorical].astype(np.int64)
                ]
            goes_left |= missing & tree["missing_left"][current]

            node[active] = np.where(goes_left, tree["left"][current], tree["right"][current])
            active = active[feature[node[active]] >= 0]

        return tree["value"][node]


class DecisionTreeC45(DecisionTree):
    """Binary decision tree using C4.5 gain ratio."""

    def _split_score(self, gain, n_left, n_right, n_total):
        # use normalized gain
        p_left = n_left / n_total
        p_right = n_right / n_total
        split_info = -(p_left * np.log(p_left) + p_right * np.log(p_right))
        return gain / split_info


class DecisionTreeCART(DecisionTree):
    """CART using Gini impurity, or variance reduction when regression=True."""

    def __init__(
        self, min_samples_split=2, max_depth=100, n_features=None, regression=False,
        ccp_alpha=0.0, categorical_features=None, random_state=None,
    ):
        super().__init__(
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            n_features=n_features,
            categorical_features=categorical_features,
            random_state=random_state,
        )
        self.regression=regression
        self.ccp_alpha = ccp_alpha

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if self.regression:
            tags.estimator_type = "regressor"
            tags.classifier_tags = None
            tags.regressor_tags = RegressorTags()
        return tags

    def score(self, X, y, sample_weight=None):
        if self.regression:
            return RegressorMixin.score(self, X, y, sample_weight=sample_weight)
        return ClassifierMixin.score(self, X, y, sample_weight=sample_weight)

    def _target_stats(self, y):
        if not self.regression:
            return super()._target_stats(y)

        if hasattr(self, "classes_"):
            del self.classes_
        y = y.astype(float)
        self._value_dtype = float
        # centering keeps cumulative variance calculations numerically stable
        self._target_offset = y.mean()
        centered = y - self._target_offset
        return np.column_stack((np.ones(y.size), centered, centered * centered))

    def _is_pure(self, node_stats, totals):
        if self.regression:
            return np.ptp(node_stats[:, 1]) == 0
        return super()._is_pure(node_stats, totals)

    def _sample_counts(self, stats):
        return stats[..., 0] if self.regression else super()._sample_counts(stats)

    def _impurity(self, stats):
        if self.regression:
            count = stats[..., 0]
            return np.maximum(stats[..., 2] / count - (stats[..., 1] / count) ** 2, 0)

        p = stats / stats.sum(axis=-1, keepdims=True)
        return 1 - (p * p).sum(axis=-1)

    def _leaf_value(self, totals):
        if self.regression:
            return float(self._target_offset + totals[1] / totals[0])
        return super()._leaf_value(totals)

    def _thresholds(self, lower, upper):
        # guard against a midpoint rounding up to the next observed value
        midpoint = lower + (upper - lower) / 2
        return np.where(midpoint < upper, midpoint, lower)

    def _category_rank(self, category_stats):
        if self.regression:
            return category_stats[:, 1] / category_stats[:, 0]
        return super()._category_rank(category_stats)

    def _prune(self, node, total_samples):
        risk = (node.n_samples / total_samples) * node.impurity
        collapsed_cost = risk + self.ccp_alpha

        if node.is_leaf_node():
            return collapsed_cost

        kept_cost = (
            self._prune(node.left, total_samples) + self._prune(node.right, total_samples)
        )

        if collapsed_cost <= kept_cost:
            # prune
            node.value = node.leaf_value
            node.left = None
            node.right = None
            node.feature = None
            node.threshold = None
            node.missing_left = False
            node.left_categories = None
            return collapsed_cost

        return kept_cost

    def fit(self, X, y):
        if self.ccp_alpha < 0:
            raise ValueError("err: alpha <0")

        super().fit(X, y)
        assert self.root is not None

        if self.ccp_alpha > 0:
            self._prune(self.root, self.root.n_samples)
            self._compile_tree()

        return self


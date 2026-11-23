import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ml_lib.model.base import BaseModel

_KERNELS = {
    "linear": "linear",
    "poly": "poly",
    "polynomial": "poly",
    "rbf": "rbf",
    "gaussian": "rbf",
}
_PREDICT_BATCH_SIZE = 1024


def _encode_binary_labels(y):
    """Return the two sorted classes and y encoded as -1/+1."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError("SVM supports one binary target. Multi-label y is not supported")

    classes, encoded = np.unique(y, return_inverse=True)
    if classes.size != 2:
        raise ValueError(f"SVM requires exactly 2 classes, got {classes.size}")
    return classes, 2.0 * encoded.ravel() - 1.0


def _signed_labels(y, classes):
    """Encode labels from an already fitted class pair as -1/+1."""
    y = np.asarray(y)
    if not np.isin(y, classes).all():
        raise ValueError(f"y must only contain the fitted classes {classes.tolist()}")
    return np.where(y == classes[1], 1.0, -1.0)


def _prepare_training_data(X, y):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinity. Please impute missing values first")

    classes, signed_y = _encode_binary_labels(y)
    if X.shape[0] != signed_y.size:
        raise ValueError(
            f"X and y have inconsistent lengths: {X.shape[0]} != {signed_y.size}"
        )
    return X, classes, signed_y


class LinearSVM(BaseModel):
    '''
        This class implements Soft Margin SVM

        Which uses the cost function as followw:
            min(w, b)[1/2 mag(w)^2 + C * {sum of max (0, 1 - y_i (wx_i + b)}]
            equation at rhs of plus sign is hinge loss function
            equation at lhs of plus sign is the minimization of w

        learning_rate is scaled by 1 / (n_samples * max(C, 1)) because the hinge
        term is a sum over all samples.
    '''

    def __init__(self, C=1, kernel='linear', degree=3, n_iters=1000, learning_rate=0.5):
        # we can seed here
        self.w = None
        self.b = None
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.n_iters=n_iters
        self.learning_rate = learning_rate
        self.history = {}


    def decision_function(self, X) -> np.ndarray:
        if self.w is None or self.b is None:
            raise RuntimeError("LinearSVM must be fitted before calling predict")
        return np.asarray(X, dtype=float) @ self.w + self.b

    def predict(self, X) -> np.ndarray:
        # these will use the final gradient descended w and b resulting from fit.
        scores = self.decision_function(X)
        return self.classes_[(scores > 0).astype(int)]

    def fit(self, X, y) -> BaseModel:
        if self.kernel != "linear":
            raise ValueError("LinearSVM only supports kernel='linear', use KernelSVM")
        if self.C <= 0:
            raise ValueError("C must be positive")

        X, self.classes_, signed_y = _prepare_training_data(X, y)
        self.w = np.random.rand(X.shape[1])
        self.b = np.random.rand()
        self.history, new_w, new_b = self._gradient_descent(X, signed_y, self.w, self.b)

        # update to the final w and b
        self.w, self.b = new_w, new_b

        return self

    def calc_hinge_loss(self, X_i, y_i, w, b):
        """Hinge loss for one sample or a batch, with y encoded as -1/+1."""
        return np.maximum(0, 1 - y_i * (X_i @ w + b))

    def calc_cost(self, X, y, w, b, C=1):
        """Primal objective, with y encoded as -1/+1."""
        margin_cost = np.dot(w, w) / 2
        return margin_cost + C * self.calc_hinge_loss(X, y, w, b).sum()

    def _calc_gradient(self, X, y, w, b):
        active = y * (X @ w + b) < 1
        y_active = y[active]

        dj_dw = w - self.C * (y_active @ X[active])
        dj_db = -self.C * y_active.sum()

        return dj_dw, dj_db

    def _gradient_descent(self, X, y, w, b):
        history = {0: self.calc_cost(X, y, w, b, self.C)}
        print(f"Iter\t0\tCost\t{history[0]}")
        step_size = self.learning_rate / (X.shape[0] * max(self.C, 1))

        for iter in range(1, self.n_iters + 1):
            dj_dw, dj_db = self._calc_gradient(X, y, w, b)
            w = w - step_size * dj_dw
            b = b - step_size * dj_db

            if iter % 100 == 0 or iter == self.n_iters:
                cost = self.calc_cost(X, y, w, b, self.C)
                print(f"Iter\t{iter}\tCost\t{cost}")
                history[iter] = cost

        return history, w, b

    # debug
    def svm_plot(self, X, y):
        xlim = np.array([X[:, 0].min()-1, X[:, 0].max()+1])
        ylim = np.array([X[:, 1].min()-1, X[:, 1].max()+1])

        # original_y_plot
        plt.figure(figsize=(4,3))
        plt.scatter(X[:, 0], X[:, 1], c=y)

        plt.title("y_true")
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])

        # y_preds
        y_preds = self.predict(X)
        plt.figure(figsize=(4,3))
        plt.scatter(X[:, 0], X[:, 1], c=y_preds)

        plt.title("y_preds")
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])


        # decision boundary plot
        xx = xlim
        yyn = (-1 -self.b - (self.w[0] * xx)) / self.w[1]  # <= -1 eq
        yym = (-self.b - (self.w[0] * xx)) / self.w[1] # == 0 eq
        yyp = (1 -self.b - (self.w[0] * xx)) / self.w[1] # >= 1 eq

        plt.plot(xx, yyn, color='red', linestyle='dotted')
        plt.plot(xx, yym, color='black')
        plt.plot(xx, yyp, color='blue', linestyle='dotted')

        # hinge loss plot
        n_examples = min(50, X.shape[0])
        X_sample = X[:n_examples]
        y_sample = _signed_labels(y[:n_examples], self.classes_)
        t = y_sample * self.decision_function(X_sample)
        loss = self.calc_hinge_loss(X_sample, y_sample, self.w, self.b)
        plt.figure(figsize=(4,3))
        plt.scatter(t, loss)
        plt.title("loss")


        plt.show()

    def plot_history(self, show=True):
        if not self.history:
            raise RuntimeError("LinearSVM must be fitted before plotting training history")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(list(self.history), list(self.history.values()), marker='o')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        ax.set_title("SVM gradient descent history")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if show:
            plt.show()
        return ax


class KernelSVM(BaseModel):
    '''
    Binary soft-margin SVM trained in the dual with Sequential Minimal
    Optimization (SMO).

    Kernels:
        linear:            x.z
        poly/polynomial:   (gamma * x.z + coef) ** degree
        rbf/gaussian:      exp(-gamma * ||x - z||^2)

    Working-set selection follows the second-order rule of Fan, Chen and Lin
    (JMLR 2005). Source: https://jmlr.org/papers/v6/fan05a.html
    Kernel rows are computed on demand and kept in an LRU cache.

    C=None trains a hard-margin SVM. n_iters and learning_rate are accepted only
    for compatibility with older notebook cells and have no effect.
    '''
    def __init__(
        self, C=1, kernel='rbf', gamma=None, degree=3, n_iters=None,
        learning_rate=None, coef=None, tol=1e-3, max_iter=None, cache_size=2000,
    ):
        # we can seed here

        # self.b = np.random.rand()
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.n_iters=n_iters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.coef = coef
        self.tol = tol
        self.max_iter = max_iter
        self.cache_size = cache_size
        # self.history = {}
        self.a: np.ndarray | None = None
        self.support_vectors: np.ndarray | None = None
        self.support_vector_labels: np.ndarray | None = None
        self.intercept: float | None = None

    def fit(self, X, y) -> BaseModel:
        self._validate_hyperparameters()
        if self.n_iters is not None or self.learning_rate is not None:
            warnings.warn(
                "KernelSVM ignores n_iters and learning_rate; use max_iter and tol",
                FutureWarning,
                stacklevel=2,
            )

        X, self.classes_, signed_y = _prepare_training_data(X, y)
        # samples and features
        _, n = X.shape

        self.kernel_ = _KERNELS[self.kernel]
        self.gamma_ = 1 / n if self.gamma is None else self.gamma
        self.coef_value_ = 1.0 if self.coef is None else self.coef
        C = np.inf if self.C is None else float(self.C)

        # langragian mult, usually denoted as alpha in equations
        a, gradient, self.n_iter_ = self._solve_smo(X, signed_y, C)

        # extract support vectors, i.e. the non-zero lagr mult
        idx = a > 0
        if not np.any(idx):
            raise RuntimeError("SVM optimization produced no support vectors")

        self.a = a[idx]
        self.support_vectors = X[idx]
        self.support_vector_labels = signed_y[idx]
        self.intercept = self._compute_intercept(a, gradient, signed_y, C)
        self.objective_ = float(0.5 * (a @ gradient) - 0.5 * a.sum())
        return self

    def _validate_hyperparameters(self):
        if self.kernel not in _KERNELS:
            raise ValueError(
                f"Unknown kernel {self.kernel!r}; expected one of {sorted(_KERNELS)}"
            )
        if self.C is not None and self.C <= 0:
            raise ValueError("C must be positive, or None for a hard margin")
        if self.gamma is not None and self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.degree < 1 or int(self.degree) != self.degree:
            raise ValueError("degree must be a positive integer")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if self.max_iter is not None and self.max_iter < 1:
            raise ValueError("max_iter must be positive or None")
        if self.cache_size < 1:
            raise ValueError("cache_size must be at least 1")

    def _solve_smo(self, X, y, C):
        """Minimize 1/2 a'Qa - e'a subject to y'a = 0 and 0 <= a <= C."""
        m = y.size
        a = np.zeros(m)
        gradient = -np.ones(m)
        kernel_row = self._kernel_row_function(X)
        diagonal = self._kernel_diagonal(X)
        limit = max(100_000, 100 * m) if self.max_iter is None else self.max_iter
        min_curvature = 1e-12
        bound_tol = 1e-12 * (C if np.isfinite(C) else 1.0)

        n_iter = 0
        while True:
            violation = -y * gradient
            below_upper = a < C
            above_lower = a > 0
            up_idx = np.flatnonzero((below_upper & (y > 0)) | (above_lower & (y < 0)))
            low_idx = np.flatnonzero((below_upper & (y < 0)) | (above_lower & (y > 0)))
            if up_idx.size == 0 or low_idx.size == 0:
                break

            # first index: the maximal KKT violator
            i = up_idx[np.argmax(violation[up_idx])]
            max_violation = violation[i]
            if max_violation - violation[low_idx].min() < self.tol:
                break
            if n_iter >= limit:
                warnings.warn(
                    f"SMO stopped after {n_iter} iterations before reaching tol",
                    ConvergenceWarning,
                    stacklevel=3,
                )
                break

            # second index: the largest second-order objective decrease
            K_i = kernel_row(i)
            candidates = low_idx[violation[low_idx] < max_violation]
            linear_term = max_violation - violation[candidates]
            curvature = np.maximum(
                diagonal[i] + diagonal[candidates] - 2 * K_i[candidates], min_curvature
            )
            j = candidates[np.argmax(linear_term * linear_term / curvature)]
            K_j = kernel_row(j)

            # analytic two-variable update, clipped to the box constraints
            pair_curvature = max(diagonal[i] + diagonal[j] - 2 * K_i[j], min_curvature)
            step = (max_violation - violation[j]) / pair_curvature
            step = min(
                step,
                C - a[i] if y[i] > 0 else a[i],
                a[j] if y[j] > 0 else C - a[j],
            )
            a[i] += y[i] * step
            a[j] -= y[j] * step
            for index in (i, j):
                if a[index] <= bound_tol:
                    a[index] = 0.0
                elif np.isfinite(C) and a[index] >= C - bound_tol:
                    a[index] = C

            gradient += step * y * (K_i - K_j)
            n_iter += 1

        return a, gradient, n_iter

    @staticmethod
    def _compute_intercept(a, gradient, y, C):
        y_gradient = y * gradient
        free = (a > 0) & (a < C)
        if free.any():
            return float(-y_gradient[free].mean())

        # without free support vectors, use the midpoint of the KKT interval
        at_upper = a >= C
        at_lower = a <= 0
        upper_mask = (at_upper & (y < 0)) | (at_lower & (y > 0))
        lower_mask = (at_upper & (y > 0)) | (at_lower & (y < 0))
        upper = y_gradient[upper_mask].min() if upper_mask.any() else np.inf
        lower = y_gradient[lower_mask].max() if lower_mask.any() else -np.inf
        return float(-(upper + lower) / 2)

    def _kernel_matrix(self, A, B):
        dots = A @ B.T
        if self.kernel_ == "linear":
            return dots
        if self.kernel_ == "poly":
            return (self.gamma_ * dots + self.coef_value_) ** self.degree

        squared = (A * A).sum(axis=1, keepdims=True) + (B * B).sum(axis=1) - 2 * dots
        return np.exp(-self.gamma_ * np.maximum(squared, 0))

    def _kernel_diagonal(self, X):
        if self.kernel_ == "rbf":
            return np.ones(X.shape[0])
        squared_norms = np.einsum("ij,ij->i", X, X)
        if self.kernel_ == "linear":
            return squared_norms
        return (self.gamma_ * squared_norms + self.coef_value_) ** self.degree

    def _kernel_row_function(self, X):
        cache = OrderedDict()
        squared_norms = np.einsum("ij,ij->i", X, X)

        def compute(i):
            dots = X @ X[i]
            if self.kernel_ == "linear":
                return dots
            if self.kernel_ == "poly":
                return (self.gamma_ * dots + self.coef_value_) ** self.degree
            squared = squared_norms + squared_norms[i] - 2 * dots
            return np.exp(-self.gamma_ * np.maximum(squared, 0))

        def row(i):
            values = cache.get(i)
            if values is None:
                values = compute(i)
                cache[i] = values
                if len(cache) > self.cache_size:
                    cache.popitem(last=False)
            else:
                cache.move_to_end(i)
            return values

        return row

    def decision_function(self, X):
        a = self.a
        support_vectors = self.support_vectors
        support_vector_labels = self.support_vector_labels
        intercept = self.intercept
        if (
            a is None
            or support_vectors is None
            or support_vector_labels is None
            or intercept is None
        ):
            raise RuntimeError("must be fitted before calling predict")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != support_vectors.shape[1]:
            raise ValueError("X must have the same number of features as the fitted model")

        weights = a * support_vector_labels
        scores = np.empty(X.shape[0])
        for start in range(0, X.shape[0], _PREDICT_BATCH_SIZE):
            stop = start + _PREDICT_BATCH_SIZE
            scores[start:stop] = self._kernel_matrix(X[start:stop], support_vectors) @ weights
        return scores + intercept

    def calc_cost(self, X, y):
        scores = self.decision_function(X)

        if self.a is None:
            raise ValueError("a must not be none")

        if self.support_vectors is None:
            raise ValueError("support vectors cannot be empty")

        signed_y = _signed_labels(y, self.classes_)
        if signed_y.shape != scores.shape:
            raise ValueError("y must contain one label per sample")

        weights = self.a * self.support_vector_labels
        support_scores = self.decision_function(self.support_vectors) - self.intercept
        margin_cost = 0.5 * (weights @ support_scores)

        return float(margin_cost + (self.C or 0) * np.maximum(0, 1 - signed_y * scores).sum())

    def predict(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[(scores > 0).astype(int)]


    # debug
    def svm_plot(self, X, y, *, grid_size=150, show=True):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have exactly two features for plotting")
        self.decision_function(X)
        padding = 0.1 * np.maximum(np.ptp(X, axis=0), 1.0)
        lower = X.min(axis=0) - padding
        upper = X.max(axis=0) + padding
        xx, yy = np.meshgrid(
            np.linspace(lower[0], upper[0], grid_size),
            np.linspace(lower[1], upper[1], grid_size),
        )
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        scores = self.decision_function(grid).reshape(xx.shape)

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(X[:, 0], X[:, 1], c=y)
        ax.contour(
            xx, yy, scores, levels=[-1, 0, 1],
            colors=['red', 'black', 'blue'],
            linestyles=['dotted', 'solid', 'dotted'], linewidths=1.5,
        )
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f"Kernel SVM · {self.kernel.upper()}")
        fig.tight_layout()
        if show:
            plt.show()
        return ax

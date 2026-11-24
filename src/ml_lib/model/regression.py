from abc import ABC, abstractmethod
from numbers import Integral
from pathlib import Path

import numpy as np

from ml_lib.model.base import BaseModel
from ml_lib.utils.functions import Sigmoid

_MAX_EXPONENT = 700.0


def _soft_threshold(w, threshold):
    return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)


class Penalty(ABC):
    """Regularization term applied to the model weights."""

    def __init__(self, alpha):
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = alpha

    @abstractmethod
    def __call__(self, w):
        """Return the penalty value at w."""

    @abstractmethod
    def grad(self, w):
        """Return a gradient or subgradient at w."""

    @abstractmethod
    def prox(self, w, step):
        """Return argmin_u [penalty(u) + ||u - w||^2 / (2 * step)]."""


class l1(Penalty):
    def __call__(self, w):
        return self.alpha * np.abs(w).sum()

    def grad(self, w):
        return self.alpha * np.sign(w)

    def prox(self, w, step):
        return _soft_threshold(w, step * self.alpha)


class l2(Penalty):
    def __call__(self, w):
        return 0.5 * self.alpha * np.dot(w, w)

    def grad(self, w):
        return self.alpha * w

    def prox(self, w, step):
        return w / (1 + step * self.alpha)


class elastic_net(Penalty):
    def __init__(self, alpha, l1_ratio=0.5):
        super().__init__(alpha)
        if not 0 <= l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1")
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        absolute = np.abs(w).sum()
        squared = 0.5 * np.dot(w, w)
        return self.alpha * (self.l1_ratio * absolute + (1 - self.l1_ratio) * squared)

    def grad(self, w):
        l1_c = self.l1_ratio * np.sign(w)
        l2_c = (1 - self.l1_ratio) * w

        return self.alpha * (l1_c + l2_c)

    def prox(self, w, step):
        thresholded = _soft_threshold(w, step * self.alpha * self.l1_ratio)
        return thresholded / (1 + step * self.alpha * (1 - self.l1_ratio))


def save_loss_video(losses, path, *, ylabel="Training log loss"):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation

    if len(losses) == 0:
        raise ValueError("Fit the model first")

    path = Path(path)
    steps = np.arange(len(losses))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    line, = ax.plot(steps, losses, marker=".", markevery=[-1])
    ax.set(xlabel="Gradient updates completed", ylabel=ylabel,
           title="Logistic regression training")
    fig.tight_layout()

    def update(end):
        line.set_data(steps[:end], losses[:end])
        return (line,)

    frames = np.linspace(1, len(losses), min(150, len(losses)), dtype=int)
    animation = FuncAnimation(fig, update, frames=frames, repeat=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation.save(path, writer=FFMpegWriter(fps=25, codec="libx264",
                                                 extra_args=["-pix_fmt", "yuv420p"]))
    finally:
        plt.close(fig)
    return path


class LogisticRegression(BaseModel):
    """
    Binary logistic regression trained with proximal gradient descent.

    penalty: None or a Penalty, such as l1, l2, or elastic_net
        applied to the weights only, never to the bias
    tol: float or None
        stop when the objective changes by at most tol, None runs all n_iters
    """

    def __init__(self, learning_rate=0.001, n_iters=1000, penalty=None, tol=None):
        self.penalty = penalty
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.tol = tol
        self.weights = None
        self.bias = None
        self.losses = []
        self.sigmoid = Sigmoid()

    def _validate_hyperparameters(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not isinstance(self.n_iters, Integral) or self.n_iters < 1:
            raise ValueError("n_iters must be a positive integer")
        if self.tol is not None and self.tol < 0:
            raise ValueError("tol must be non-negative or None")
        if self.penalty is not None and not isinstance(self.penalty, Penalty):
            raise TypeError("penalty must be None or a Penalty instance")

    def decision_function(self, X):
        if self.weights is None or self.bias is None:
            raise RuntimeError("LogisticRegression must be fitted before calling predict")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.weights.size:
            raise ValueError(f"X must have shape (n_samples, {self.weights.size}), got {X.shape}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity. Please impute missing values first")
        return (X @ self.weights) + self.bias

    def feed_forward(self, X):
        return self.sigmoid(self.decision_function(X))

    def fit(self, X, y) -> BaseModel:
        self._validate_hyperparameters()
        X, target = self._prepare_training_data(X, y)
        n_samples, n_features = X.shape
        signs = 2.0 * target - 1.0

        weights = np.zeros(n_features)
        bias = 0.0
        self.losses = []
        self.n_iter_ = 0

        for _ in range(self.n_iters):
            # log loss is log(1 + exp(-sign * score))
            # one exp serves loss and gradient
            exponent = -signs * (X @ weights + bias)
            odds = np.exp(np.minimum(exponent, _MAX_EXPONENT))
            loss = float((np.log1p(odds) + np.maximum(exponent - _MAX_EXPONENT, 0)).mean())
            if self.penalty is not None:
                loss += float(self.penalty(weights))

            if self.tol is not None and self.losses and abs(self.losses[-1] - loss) <= self.tol:
                self.losses.append(loss)
                break
            self.losses.append(loss)

            # gradients
            residual = -signs * odds / (1 + odds)  # sigmoid(score) - target
            dw = (X.T @ residual) / n_samples
            db = residual.mean()

            weights = self._update_weights(weights, dw)
            bias -= self.learning_rate * db
            self.n_iter_ += 1

        self.weights = weights
        self.bias = bias
        return self

    def _prepare_training_data(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError("LogisticRegression supports one binary target")
        if X.shape[0] != y.size:
            raise ValueError(f"X and y have inconsistent lengths: {X.shape[0]} != {y.size}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity. Please impute missing values first")

        self.classes_, encoded = np.unique(y, return_inverse=True)
        if self.classes_.size != 2:
            raise ValueError(
                f"LogisticRegression requires exactly 2 classes, got {self.classes_.size}"
            )
        return X, encoded.ravel().astype(float)

    def _update_weights(self, weights, gradient):
        step = weights - self.learning_rate * gradient
        if self.penalty is None:
            return step
        return self.penalty.prox(step, self.learning_rate)

    def save_loss_video(self, path):
        """Export this fitted model's loss history as an MP4."""
        label = "Training log loss" if self.penalty is None else "Log loss + regularization penalty"
        return save_loss_video(self.losses, path, ylabel=label)

    def predict(self, X):
        # sigmoid(score) > 0.5 is the same as score > 0
        scores = self.decision_function(X)
        return self.classes_[(scores > 0).astype(int)]


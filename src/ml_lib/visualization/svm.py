"""SVM decision boundaries, margins, and support vectors."""

from __future__ import annotations

from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ml_lib.model.svm import KernelSVM, LinearSVM, _encode_binary_labels, _signed_labels

from .regression import _sample_rows
from .style import CLASS_COLORS, INK, NEUTRAL_COLOR, SECONDARY_INK, SURFACE

_COMPARED_KERNELS = ("linear", "poly", "rbf")
_KERNEL_NAMES = {"linear": "Linear", "poly": "Polynomial", "rbf": "Gaussian"}
_CLASS_MARKERS = ("o", "s")


def plot_svm(model, X, y, *, grid_size=150, ax=None, show=True):
    """Draw a two-feature SVM's boundary, margins, and support vectors."""
    if not isinstance(model, (LinearSVM, KernelSVM)):
        raise TypeError("model must be a LinearSVM or KernelSVM")
    _validate_positive(grid_size, "grid_size", minimum=2)

    names = _column_names(X)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("plot_svm needs two features, use plot_kernel_comparison for more")
    scores = model.decision_function(X)
    signs = _signed_labels(_as_label_vector(y, X.shape[0]), model.classes_)
    support = (
        model.support_vectors if isinstance(model, KernelSVM) else X[signs * scores <= 1]
    )

    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=(7, 5.5), facecolor=SURFACE, layout="constrained")
    _draw_panel(ax, model, X, signs, support, grid_size)
    ax.set_xlabel(names[0], color=INK)
    ax.set_ylabel(names[1], color=INK)
    ax.legend(
        handles=_legend_handles(model.classes_), loc="upper center",
        bbox_to_anchor=(0.5, -0.11), ncols=5, frameon=False, fontsize=8,
    )
    if show and owns_figure:
        plt.show()
    return ax


def plot_svm_history(model, ax=None, show=True):
    """Draw the LinearSVM training cost."""
    if not isinstance(model, LinearSVM):
        raise TypeError("Only LinearSVM records a training cost history")
    if not model.history:
        raise RuntimeError("LinearSVM must be fitted before plotting training history")

    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)
    ax.plot(
        list(model.history), list(model.history.values()),
        color=CLASS_COLORS[0], linewidth=2, marker="o", markersize=6,
    )
    ax.set_xlabel("Iteration", color=INK)
    ax.set_ylabel("Cost", color=INK)
    ax.set_title("Training cost", loc="left", fontsize=10, color=INK)
    ax.tick_params(colors=SECONDARY_INK, labelsize=8)
    ax.grid(color=NEUTRAL_COLOR, alpha=0.2, linewidth=0.6)
    if show and owns_figure:
        plt.show()
    return ax


def plot_kernel_comparison(X, y, *, max_samples=1000, grid_size=150, show=True):
    """
    Refit linear, polynomial, and Gaussian SVMs in 2D and draw them side by side.

    Features are standardized, and more than two are reduced with PCA. These
    panels explain kernel behavior, not the performance of a full-feature model.
    """
    _validate_positive(max_samples, "max_samples", minimum=2)
    _validate_positive(grid_size, "grid_size", minimum=2)

    names = _column_names(X)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("X must have at least two features")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinity. Please impute missing values first")
    y = _as_label_vector(y, X.shape[0])
    classes, signs = _encode_binary_labels(y)

    points, labels, note = _project(X, names)
    rows = _sample_rows(signs, max_samples)
    if rows.size < X.shape[0]:
        note += f", {rows.size:,} of {X.shape[0]:,} rows"

    # scaling gamma to the data keeps the polynomial solve well-conditioned on outliers
    variance = points[rows].var()
    gamma = 1 / (2 * variance) if variance > 0 else None

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 5.2), sharex=True, sharey=True,
        facecolor=SURFACE, layout="constrained",
    )
    for ax, kernel in zip(axes, _COMPARED_KERNELS):
        model = KernelSVM(kernel=kernel, gamma=gamma).fit(points[rows], y[rows])
        _draw_panel(ax, model, points[rows], signs[rows], model.support_vectors, grid_size)
        ax.set_xlabel(labels[0], color=INK)
    axes[0].set_ylabel(labels[1], color=INK)
    fig.suptitle(note, x=0.01, ha="left", fontsize=11, color=INK)
    fig.legend(
        handles=_legend_handles(classes), loc="outside lower center",
        ncols=5, frameon=False, fontsize=9,
    )
    if show:
        plt.show()
    return axes


def _validate_positive(value, name, minimum):
    if not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _column_names(X):
    columns = getattr(X, "columns", None)
    if columns is None:
        return [f"X[{i}]" for i in range(np.shape(X)[1])] if np.ndim(X) == 2 else []
    return [str(column) for column in columns]


def _as_label_vector(y, n_rows):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1 or y.size != n_rows:
        raise ValueError("y must be one label per row of X")
    return y


def _project(X, names):
    """Return 2D points, their axis labels, and a short caption."""
    scaled = StandardScaler().fit_transform(X)
    if X.shape[1] == 2:
        return scaled, [f"{name} (standardized)" for name in names], "Standardized features"

    pca = PCA(n_components=2, random_state=0)
    points = pca.fit_transform(scaled)
    variance = pca.explained_variance_ratio_
    labels = [f"PC{i + 1} ({share:.1%})" for i, share in enumerate(variance)]
    return points, labels, f"Refit on 2D PCA, {variance.sum():.1%} variance"


def _model_name(model):
    if isinstance(model, LinearSVM):
        return "Linear SVM"
    name = _KERNEL_NAMES[model.kernel_]
    return f"{name} (degree {model.degree})" if model.kernel_ == "poly" else name


def _draw_panel(ax, model, X, signs, support, grid_size):
    ptp = np.ptp(X, axis=0)
    padding = np.where(ptp > 0, 0.08 * ptp, 0.5)
    lower, upper = X.min(axis=0) - padding, X.max(axis=0) + padding
    xx, yy = np.meshgrid(
        np.linspace(lower[0], upper[0], grid_size),
        np.linspace(lower[1], upper[1], grid_size),
    )
    scores = model.decision_function(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    low, high = scores.min(), scores.max()

    ax.set_facecolor(SURFACE)
    ax.contourf(
        xx, yy, scores, levels=[min(low, -1.0) - 1, 0, max(high, 1.0) + 1],
        colors=CLASS_COLORS[:2], alpha=0.14, zorder=0,
    )
    for sign, color, marker in zip((-1, 1), CLASS_COLORS, _CLASS_MARKERS):
        points = X[signs == sign]
        ax.scatter(
            points[:, 0], points[:, 1], s=16, c=color, marker=marker,
            edgecolors=SURFACE, linewidths=0.3, alpha=0.8, zorder=1,
        )
    ax.scatter(
        support[:, 0], support[:, 1], s=40, facecolors="none",
        edgecolors=INK, linewidths=0.45, alpha=0.45, zorder=2,
    )
    for level, color, style, width in (
        (-1, SECONDARY_INK, "--", 1.3), (1, SECONDARY_INK, "--", 1.3), (0, INK, "-", 1.8),
    ):
        if low < level < high:
            ax.contour(
                xx, yy, scores, levels=[level], colors=color,
                linestyles=style, linewidths=width, zorder=3,
            )
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.tick_params(colors=SECONDARY_INK, labelsize=8)
    ax.set_title(
        f"{_model_name(model)}: {support.shape[0]:,} support vectors",
        loc="left", fontsize=10, color=INK,
    )


def _legend_handles(classes):
    handles = [
        Line2D([], [], marker=marker, linestyle="", markersize=6, color=color,
               label=f"Class {label}")
        for label, color, marker in zip(classes, CLASS_COLORS, _CLASS_MARKERS)
    ]
    return [
        *handles,
        Line2D([], [], color=INK, linewidth=1.8, label="Boundary"),
        Line2D([], [], color=SECONDARY_INK, linewidth=1.3, linestyle="--", label="Margin"),
        Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor="none",
               markeredgecolor=INK, markeredgewidth=0.7, label="Support vector"),
    ]

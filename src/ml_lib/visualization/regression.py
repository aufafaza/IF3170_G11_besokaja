"""Logistic regression loss contours and training video."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize

from ml_lib.model.regression import _MAX_EXPONENT

from .style import CLASS_COLORS, INK, NEUTRAL_COLOR, SECONDARY_INK, SURFACE

_LOSS_CMAP = LinearSegmentedColormap.from_list("loss", ["#9cc3ee", "#d8e7f7", SURFACE])
_PATH_CMAP = LinearSegmentedColormap.from_list("path", ["#f0915f", CLASS_COLORS[1], "#6f2608"])
_N_LEVELS = 12
_LABEL_EDGE_MARGIN = 0.06


def plot_loss_contour(
    model, X, y, feature=None, grid_size=100, max_rows=5000, ax=None, show=True
):
    """
    Draw loss contours over (bias, feature weight) with the training path.

    feature defaults to the largest absolute final weight. Other weights stay
    at their final values.
    """
    history = getattr(model, "parameter_history_", None)
    if history is None:
        raise RuntimeError("Fit with track_parameters=True before plotting the loss contour")
    if not isinstance(grid_size, Integral) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least 2")
    if max_rows is not None and (not isinstance(max_rows, Integral) or max_rows < 1):
        raise ValueError("max_rows must be a positive integer or None")

    columns = getattr(X, "columns", None)
    X = np.asarray(X, dtype=float)
    scores = model.decision_function(X)
    signs = _class_signs(model, y, X.shape[0])
    index, name = _resolve_feature(model, feature, columns)

    rows = _sample_rows(signs, max_rows)
    values = X[rows, index]
    other_scores = scores[rows] - model.bias - model.weights[index] * values
    path = history[:, [0, index + 1]]
    theta0 = _grid_axis(path[:, 0], grid_size)
    theta1 = _grid_axis(path[:, 1], grid_size)
    surface = _loss_surface(model, other_scores, values, signs[rows], index, theta0, theta1)

    owns_figure = ax is None
    if owns_figure:
        _, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)

    _draw_contours(ax, theta0, theta1, surface)
    _draw_path(ax, path)
    ax.set_xlim(theta0[0], theta0[-1])
    ax.set_ylim(theta1[0], theta1[-1])
    ax.set_xlabel("θ₀ (bias)", color=INK)
    ax.set_ylabel(f"θ₁ (weight of {name})", color=INK)
    ax.tick_params(colors=SECONDARY_INK, labelsize=8)
    ax.set_title(_title(model, rows.size, X.shape[0]), loc="left", fontsize=10, color=INK)
    if show and owns_figure:
        plt.show()
    return ax


def save_loss_video(model, path):
    """Export a fitted model's loss history as an MP4."""
    losses = model.losses
    if len(losses) == 0:
        raise ValueError("Fit the model first")

    ylabel = "Training log loss" if model.penalty is None else "Log loss and regularization penalty"
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


def _class_signs(model, y, n_rows):
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1 or y.size != n_rows:
        raise ValueError("y must be one label per row of X")
    if not np.isin(y, model.classes_).all():
        raise ValueError("y contains labels the model was not trained on")
    return np.where(y == model.classes_[1], 1.0, -1.0)


def _resolve_feature(model, feature, columns):
    n_features = model.weights.size
    if feature is None:
        index = int(np.argmax(np.abs(model.weights)))
    elif isinstance(feature, Integral) and not isinstance(feature, bool):
        index = int(feature)
        if not 0 <= index < n_features:
            raise ValueError(f"feature index must be between 0 and {n_features - 1}")
    elif columns is not None and feature in columns:
        index = list(columns).index(feature)
    else:
        raise ValueError(f"Unknown feature: {feature!r}")
    return index, str(columns[index]) if columns is not None else f"X[{index}]"


def _sample_rows(signs, max_rows):
    """Pick a repeatable sample that keeps the class ratio."""
    n_rows = signs.size
    if max_rows is None or n_rows <= max_rows:
        return np.arange(n_rows)
    shuffled = np.random.default_rng(0).permutation(n_rows)
    by_class = shuffled[np.argsort(signs[shuffled], kind="stable")]
    return by_class[np.linspace(0, n_rows - 1, max_rows, dtype=int)]


def _grid_axis(values, grid_size):
    low, high = values.min(), values.max()
    pad = max(0.2 * (high - low), 0.05)
    return np.linspace(low - pad, high + pad, grid_size)


def _loss_surface(model, other_scores, values, signs, index, theta0, theta1):
    """Return the objective with rows for theta1 and columns for theta0."""
    weights = model.weights.copy()
    surface = np.empty((theta1.size, theta0.size))
    bias_term = signs[:, None] * theta0
    for row, weight in enumerate(theta1):
        margin = signs * (other_scores + weight * values)
        exponent = -(margin[:, None] + bias_term)
        odds = np.exp(np.minimum(exponent, _MAX_EXPONENT))
        surface[row] = (np.log1p(odds) + np.maximum(exponent - _MAX_EXPONENT, 0)).mean(axis=0)
        if model.penalty is not None:
            weights[index] = weight
            surface[row] += float(model.penalty(weights))
    return surface


def _draw_contours(ax, theta0, theta1, surface):
    # quantile levels keep detail near the minimum, where the loss is flat
    levels = np.unique(np.quantile(surface, np.linspace(0.03, 0.97, _N_LEVELS)))
    if levels.size < 2:
        return
    ax.contourf(theta0, theta1, surface, levels=levels, cmap=_LOSS_CMAP, extend="both", zorder=0)
    lines = ax.contour(
        theta0, theta1, surface, levels=levels,
        colors=NEUTRAL_COLOR, linewidths=0.6, zorder=1,
    )
    labels = ax.clabel(lines, levels[::2], fmt="%.4g", fontsize=7, colors=SECONDARY_INK)

    # labels at the plot edge get cut off
    margin_x = _LABEL_EDGE_MARGIN * (theta0[-1] - theta0[0])
    margin_y = _LABEL_EDGE_MARGIN * (theta1[-1] - theta1[0])
    for label in labels:
        x, y = label.get_position()
        inside = theta0[0] + margin_x < x < theta0[-1] - margin_x
        label.set_visible(inside and theta1[0] + margin_y < y < theta1[-1] - margin_y)


def _draw_path(ax, path):
    ax.plot(path[:, 0], path[:, 1], color=SURFACE, linewidth=4, zorder=2)
    segments = np.stack([path[:-1], path[1:]], axis=1)
    n_updates = len(segments)
    line = LineCollection(
        segments, cmap=_PATH_CMAP, norm=Normalize(0, n_updates), linewidth=2, zorder=3,
    )
    line.set_array(np.arange(1, n_updates + 1))
    ax.add_collection(line)
    ax.figure.colorbar(line, ax=ax, label="Gradient update", format="%d", pad=0.02)

    ax.plot(*path[0], marker="o", markersize=8, markerfacecolor=SURFACE,
            markeredgecolor=INK, linestyle="", zorder=4)
    ax.plot(*path[-1], marker="*", markersize=13, markerfacecolor=INK,
            markeredgecolor=SURFACE, linestyle="", zorder=4)
    _label_endpoint(ax, path[0], path[-1], "Start")
    _label_endpoint(ax, path[-1], path[0], "Final")


def _label_endpoint(ax, point, other, text):
    # the side facing away from the other endpoint stays clear of the path
    away = np.where(point >= other, 1, -1)
    ax.annotate(
        text, point, xytext=(10 * away[0], 10 * away[1]), textcoords="offset points",
        ha="left" if away[0] > 0 else "right", va="bottom" if away[1] > 0 else "top",
        fontsize=8, color=INK, zorder=5,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": SURFACE,
              "edgecolor": "none", "alpha": 0.85},
    )


def _title(model, n_used, n_rows):
    title = "Log loss" if model.penalty is None else "Log loss with penalty"
    notes = []
    if model.weights.size > 1:
        notes.append("slice at final weights")
    if n_used < n_rows:
        notes.append(f"{n_used:,} of {n_rows:,} rows")
    return f"{title}\n{', '.join(notes)}" if notes else title

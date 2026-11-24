"""Naive Bayes feature-class graph and conditional probabilities."""

from __future__ import annotations

from itertools import pairwise
from numbers import Integral

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ml_lib.model.bayes import DiscreteNB

from .style import CLASS_COLORS, INK, NEUTRAL_COLOR, SECONDARY_INK, SURFACE
from .tree import _format_number

_CLASS_X = 0.2
_FEATURE_X = 0.62
_EDGE_LABEL_POSITION = 0.74
_MIN_EDGE_WIDTH = 0.8
_MAX_EDGE_WIDTH = 5.0
_MAX_NAME_LENGTH = 38
_MAX_FIGURE_HEIGHT = 20.0


def plot_bayes_network(model, top_n=10, ax=None, show=True):
    """
    Draw the class-to-feature graph of a fitted DiscreteNB.

    Edges show mutual information in bits. top_n keeps the strongest features
    (None shows all).
    """
    _check_model(model)
    if top_n is not None and (not isinstance(top_n, Integral) or top_n < 1):
        raise ValueError("top_n must be a positive integer or None")

    names = _feature_names(model)
    information = np.array([_mutual_information(c) for c in model.category_count_])
    order = np.argsort(-information, kind="stable")[:top_n]
    n_shown = order.size

    owns_figure = ax is None
    if owns_figure:
        height = min(max(3.5, 0.38 * n_shown + 1.2), _MAX_FIGURE_HEIGHT)
        _, ax = plt.subplots(figsize=(9, height), facecolor=SURFACE, layout="constrained")

    graph = nx.DiGraph()
    positions = {"class": (_CLASS_X, (n_shown - 1) / 2)}
    for rank, index in enumerate(order):
        node = int(index)
        positions[node] = (_FEATURE_X, n_shown - 1 - rank)
        graph.add_edge("class", node, information=information[index])

    edges = list(graph.edges)
    strongest = information[order[0]]
    scale = information[order] / strongest if strongest > 0 else np.zeros(n_shown)
    nx.draw_networkx_edges(
        graph, positions, ax=ax, edgelist=edges,
        width=list(_MIN_EDGE_WIDTH + (_MAX_EDGE_WIDTH - _MIN_EDGE_WIDTH) * scale),
        edge_color=CLASS_COLORS[0], alpha=0.75, arrows=True, arrowstyle="-|>", arrowsize=10,
        node_size=0, min_source_margin=4, min_target_margin=6,
    )

    box = {"boxstyle": "round,pad=0.35", "facecolor": SURFACE, "edgecolor": NEUTRAL_COLOR}
    ax.text(
        *positions["class"], _class_label(model), ha="right", va="center", ma="center",
        fontsize=9, color=INK, zorder=4, bbox=box,
    )
    for _, node in edges:
        x, y = positions[node]
        ax.text(
            x, y, _shorten(names[node]), ha="left", va="center",
            fontsize=9, color=INK, zorder=4, bbox=box,
        )
        ax.text(
            _CLASS_X + _EDGE_LABEL_POSITION * (x - _CLASS_X),
            positions["class"][1] + _EDGE_LABEL_POSITION * (y - positions["class"][1]),
            f"{information[node]:.3f}", ha="center", va="center", fontsize=7,
            color=SECONDARY_INK, zorder=3,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1},
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, n_shown - 0.3)
    ax.set_axis_off()
    ax.set_title(
        f"Naive Bayes network\n{_network_note(n_shown, len(names))}",
        loc="left", fontsize=10, color=INK,
    )
    if show and owns_figure:
        plt.show()
    return ax


def plot_feature_class(model, feature, ax=None, show=True):
    """Draw the smoothed P(value | class) of one feature, by name or index."""
    _check_model(model)
    n_classes = model.classes_.size
    if n_classes > len(CLASS_COLORS):
        raise ValueError(f"plot_feature_class supports at most {len(CLASS_COLORS)} classes")

    index = _resolve_feature(model, feature)
    labels = _value_labels(model, index)
    order = _display_order(model, index, labels)
    probabilities = np.exp(model.feature_log_prob_[index])[:, order]
    labels = [labels[i] for i in order]
    positions = np.arange(len(labels))

    owns_figure = ax is None
    if owns_figure:
        height = min(max(3.5, 0.42 * len(labels) + 1.4), _MAX_FIGURE_HEIGHT)
        _, ax = plt.subplots(figsize=(8, height), facecolor=SURFACE, layout="constrained")
    ax.set_facecolor(SURFACE)

    bar_height = 0.8 / n_classes
    for c, (label, color) in enumerate(zip(model.classes_, CLASS_COLORS)):
        offset = (c - (n_classes - 1) / 2) * bar_height
        ax.barh(
            positions + offset, probabilities[c], height=bar_height, color=color,
            edgecolor=SURFACE, linewidth=1, label=f"Class {label}", zorder=2,
        )

    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Probability", color=INK)
    ax.tick_params(colors=SECONDARY_INK, labelsize=8)
    ax.grid(axis="x", color=NEUTRAL_COLOR, alpha=0.25, linewidth=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        f"P({_feature_names(model)[index]} | class)", loc="left", fontsize=10, color=INK,
    )
    ax.legend(
        loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=min(n_classes, 4),
        frameon=False, fontsize=8,
    )
    if show and owns_figure:
        plt.show()
    return ax


def _check_model(model):
    if not isinstance(model, DiscreteNB):
        raise TypeError("model must be a DiscreteNB")
    if not hasattr(model, "category_count_"):
        raise RuntimeError("DiscreteNB must be fitted before plotting")


def _feature_names(model):
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return [f"X[{i}]" for i in range(model.n_features_in_)]
    return [str(name) for name in names]


def _resolve_feature(model, feature):
    names = getattr(model, "feature_names_in_", None)
    if names is not None and feature in list(names):
        return list(names).index(feature)
    if (
        isinstance(feature, Integral)
        and not isinstance(feature, bool)
        and 0 <= feature < model.n_features_in_
    ):
        return int(feature)
    raise ValueError(f"Unknown feature: {feature!r}")


def _mutual_information(counts):
    """Return I(feature, class) in bits from a class-by-value count table."""
    joint = counts / counts.sum()
    independent = joint.sum(axis=0, keepdims=True) * joint.sum(axis=1, keepdims=True)
    seen = joint > 0
    return max(float((joint[seen] * np.log2(joint[seen] / independent[seen])).sum()), 0.0)


def _value_labels(model, index):
    kind = model.feature_kinds_[index]
    if kind == "categorical":
        labels = [_shorten(str(value)) for value in model._encoder.categories_[index]]
    elif kind == "discrete":
        labels = [str(int(value)) for value in model.discrete_values_[index]]
    else:
        labels = _bin_labels(model.bin_edges_[index])
    if model.missing_codes_[index] >= 0:
        labels.append("Missing")
    return labels


def _display_order(model, index, labels):
    """Sort categories by name and keep Missing last."""
    if model.feature_kinds_[index] != "categorical":
        return np.arange(len(labels))
    n_categories = len(model._encoder.categories_[index])
    named = sorted(range(n_categories), key=labels.__getitem__)
    return np.array([*named, *range(n_categories, len(labels))])


def _bin_labels(edges):
    if edges.size == 0:
        return ["All values"]
    text = [_format_number(edge) for edge in edges]
    middle = [f"[{low}, {high})" for low, high in pairwise(text)]
    return [f"< {text[0]}", *middle, f"≥ {text[-1]}"]


def _class_label(model):
    priors = [
        f"P({label}) = {prior:.3f}" for label, prior in zip(model.classes_, model.class_prior_)
    ]
    return "\n".join(["Class", *priors])


def _network_note(n_shown, n_features):
    if n_shown < n_features:
        return f"top {n_shown} of {n_features} features, mutual information (bits)"
    return f"{n_features} features, mutual information (bits)"


def _shorten(text):
    if len(text) <= _MAX_NAME_LENGTH:
        return text
    return f"{text[:_MAX_NAME_LENGTH - 1]}…"

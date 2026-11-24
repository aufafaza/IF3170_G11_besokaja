from __future__ import annotations

import heapq
import textwrap
import warnings
from itertools import count
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

from ml_lib.model.dtl import DecisionTreeCART

_CLASS_COLORS = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948",
)
_NEUTRAL_COLOR = "#898781"
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_SECONDARY_INK = "#52514e"

_BASE_FONT_SIZE = 8.5
_MIN_READABLE_FONT_SIZE = 5.0
_MAX_CATEGORY_LINE = 22
_BOX_PAD = 0.35
_SLOT_GAP = 0.18
_ROW_GAP = 0.42
_LEGEND_HEIGHT = 0.4
_MIN_FIGURE_WIDTH = 6.0
_MIN_FIGURE_HEIGHT = 3.0
_MAX_FIGURE_WIDTH = 40.0
_MAX_FIGURE_HEIGHT = 24.0


def plot_tree(tree, feature_names=None, max_depth=3, top_n=None, ax=None, show=True):
    """
    Draw a fitted decision tree.

    max_depth limits how deep branches can be expanded (None for no limit).
    top_n limits the number of splits shown, choosing the largest weighted
    impurity decreases while keeping the shown tree connected to the root.
    """
    if tree.root is None:
        raise RuntimeError("DecisionTree must be fitted before plotting")
    _validate_limits(max_depth, top_n)
    names = _resolve_feature_names(tree, feature_names)

    selected = _select_splits(tree.root, max_depth, top_n)
    nodes, n_slots, n_rows = _layout(tree.root, selected)

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(_MIN_FIGURE_WIDTH, _MIN_FIGURE_HEIGHT), facecolor=_SURFACE)
    else:
        fig = ax.figure

    ax.set_xlim(-0.5, n_slots - 0.5)
    ax.set_ylim(-(n_rows - 1) - 0.5, 0.5)
    ax.set_axis_off()

    texts = [
        _draw_node(ax, tree, node, x, depth, names, selected)
        for node, x, depth in nodes
    ]
    handles = _legend_handles(tree, nodes, selected)
    font_size = _fit_text(fig, ax, texts, n_slots, n_rows, owns_figure, bool(handles))
    _draw_edges(ax, nodes, selected, font_size)

    if handles:
        ax.legend(
            handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
            ncols=len(handles), frameon=False, fontsize=max(font_size, 7),
        )
    if show and owns_figure:
        plt.show()
    return ax


def _validate_limits(max_depth, top_n):
    if max_depth is not None and (not isinstance(max_depth, Integral) or max_depth < 0):
        raise ValueError("max_depth must be a nonnegative integer or None")
    if top_n is not None and (not isinstance(top_n, Integral) or top_n < 1):
        raise ValueError("top_n must be a positive integer or None")


def _resolve_feature_names(tree, feature_names):
    if feature_names is None:
        feature_names = getattr(tree, "feature_names_in_", None)
    if feature_names is None:
        return [f"X[{i}]" for i in range(tree.n_features_in_)]

    names = [str(name) for name in feature_names]
    if len(names) != tree.n_features_in_:
        raise ValueError("Provide one feature name per training column")
    return names


def _impurity_decrease(node, n_total):
    left, right = node.left, node.right
    return (
        node.n_samples * node.impurity
        - left.n_samples * left.impurity
        - right.n_samples * right.impurity
    ) / n_total


def _select_splits(root, max_depth, top_n):
    """Return ids of split nodes to expand, chosen best-first from the root."""
    selected = set()
    order = count()
    candidates = []

    def add_candidate(node, depth):
        if node.is_leaf_node() or (max_depth is not None and depth >= max_depth):
            return
        priority = -_impurity_decrease(node, root.n_samples)
        heapq.heappush(candidates, (priority, next(order), depth, node))

    add_candidate(root, 0)
    while candidates and (top_n is None or len(selected) < top_n):
        _, _, depth, node = heapq.heappop(candidates)
        selected.add(id(node))
        add_candidate(node.left, depth + 1)
        add_candidate(node.right, depth + 1)
    return selected


def _layout(root, selected):
    """Give each visible leaf one slot and center each parent over its children."""
    nodes = []
    next_slot = 0
    deepest = 0

    def place(node, depth):
        nonlocal next_slot, deepest
        deepest = max(deepest, depth)
        index = len(nodes)
        nodes.append(None)
        if id(node) in selected:
            x = (place(node.left, depth + 1) + place(node.right, depth + 1)) / 2
        else:
            x = float(next_slot)
            next_slot += 1
        nodes[index] = (node, x, depth)
        return x

    place(root, 0)
    return nodes, next_slot, deepest + 1


def _format_number(value):
    if abs(value) >= 1e6:
        return f"{value:,.0f}"
    if abs(value) >= 1000:
        return f"{value:,.1f}".removesuffix(".0")
    return f"{value:.4g}"


def _split_lines(tree, node, names):
    name = names[node.feature]
    if node.left_categories is not None:
        categories = tree._encoder.categories_[node.feature][node.left_categories]
        shown = [str(category) for category in categories[:3]]
        if len(categories) > 3:
            shown.append(f"+{len(categories) - 3}")
        return [name, *textwrap.wrap(f"in {{{', '.join(shown)}}}", _MAX_CATEGORY_LINE)]
    if np.isinf(node.threshold):
        return [name, "is present"]
    return [name, f"≤ {_format_number(node.threshold)}"]


def _count_splits(node):
    if node.is_leaf_node():
        return 0
    return 1 + _count_splits(node.left) + _count_splits(node.right)


def _criterion_name(tree):
    if getattr(tree, "regression", False):
        return "variance"
    return "gini" if isinstance(tree, DecisionTreeCART) else "entropy"


def _value_text(tree, node):
    if getattr(tree, "regression", False):
        return _format_number(node.leaf_value)
    return str(node.leaf_value)


def _node_label(tree, node, names, selected):
    impurity = (
        f"{node.impurity:.3f}" if node.impurity < 1000 else _format_number(node.impurity)
    )
    summary = [f"n = {node.n_samples:,}", f"{_criterion_name(tree)} = {impurity}"]
    value = _value_text(tree, node)
    majority = "mean" if getattr(tree, "regression", False) else "majority"

    if id(node) in selected:
        lines = _split_lines(tree, node, names)
        if node.left_categories is not None or not np.isinf(node.threshold):
            lines.append(f"missing → {'Yes' if node.missing_left else 'No'}")
        lines += [f"{majority} {value}", *summary]
    elif node.is_leaf_node():
        lines = [f"predict {value}", *summary]
    else:
        hidden = _count_splits(node)
        lines = [
            f"{majority} {value}",
            f"+{hidden} hidden split{'s' if hidden != 1 else ''}",
            *summary,
        ]
    return "\n".join(lines)


def _node_color(tree, node):
    classes = getattr(tree, "classes_", None)
    if classes is None or classes.size > len(_CLASS_COLORS):
        return _NEUTRAL_COLOR
    return _CLASS_COLORS[int(np.flatnonzero(classes == node.leaf_value)[0])]


def _tint(color, strength=0.22):
    surface = np.array(to_rgb(_SURFACE))
    return tuple(strength * np.array(to_rgb(color)) + (1 - strength) * surface)


def _is_collapsed(node, selected):
    return not node.is_leaf_node() and id(node) not in selected


def _draw_node(ax, tree, node, x, depth, names, selected):
    color = _node_color(tree, node)
    return ax.text(
        x, -depth, _node_label(tree, node, names, selected),
        ha="center", va="center", color=_INK, fontsize=_BASE_FONT_SIZE, zorder=2,
        bbox={
            "boxstyle": f"round,pad={_BOX_PAD}",
            "facecolor": _tint(color),
            "edgecolor": color,
            "linewidth": 1.2,
            "linestyle": "--" if _is_collapsed(node, selected) else "-",
        },
    )


def _fit_text(fig, ax, texts, n_slots, n_rows, owns_figure, has_legend):
    """Size the figure, or shrink text, so node boxes cannot overlap."""
    extents = [text.get_window_extent() for text in texts]
    padding = 2 * _BOX_PAD * _BASE_FONT_SIZE / 72
    box_width = max(extent.width for extent in extents) / fig.dpi + padding
    box_height = max(extent.height for extent in extents) / fig.dpi + padding
    needed_width = n_slots * (box_width + _SLOT_GAP)
    needed_height = n_rows * (box_height + _ROW_GAP)

    if owns_figure:
        legend_height = _LEGEND_HEIGHT if has_legend else 0.0
        width = min(max(needed_width, _MIN_FIGURE_WIDTH), _MAX_FIGURE_WIDTH)
        height = min(
            max(needed_height + legend_height, _MIN_FIGURE_HEIGHT), _MAX_FIGURE_HEIGHT
        )
        fig.set_size_inches(width, height)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1 - legend_height / height)

    position = ax.get_position()
    figure_width, figure_height = fig.get_size_inches()
    scale = min(
        1.0,
        position.width * figure_width / needed_width,
        position.height * figure_height / needed_height,
    )
    font_size = _BASE_FONT_SIZE * scale
    if scale < 1.0:
        for text in texts:
            text.set_fontsize(font_size)
    if font_size < _MIN_READABLE_FONT_SIZE:
        warnings.warn(
            "Too many visible tree nodes for readable labels; use top_n or max_depth",
            UserWarning,
            stacklevel=3,
        )
    return font_size


def _draw_edges(ax, nodes, selected, font_size):
    positions = {id(node): x for node, x, _ in nodes}
    for node, x, depth in nodes:
        if id(node) not in selected:
            continue
        for child, answer in ((node.left, "Yes"), (node.right, "No")):
            child_x = positions[id(child)]
            ax.plot(
                [x, child_x], [-depth, -depth - 1],
                color=_NEUTRAL_COLOR, linewidth=1.0, zorder=1,
            )
            ax.text(
                (x + child_x) / 2, -depth - 0.5, answer,
                ha="center", va="center", color=_SECONDARY_INK, fontsize=font_size * 0.9,
                zorder=3, bbox={"facecolor": _SURFACE, "edgecolor": "none", "pad": 1},
            )


def _legend_handles(tree, nodes, selected):
    handles = []
    classes = getattr(tree, "classes_", None)
    if classes is not None and 2 <= classes.size <= len(_CLASS_COLORS):
        handles = [
            Patch(facecolor=_tint(color), edgecolor=color, label=f"Class {label}")
            for label, color in zip(classes, _CLASS_COLORS)
        ]
    if any(_is_collapsed(node, selected) for node, _, _ in nodes):
        handles.append(
            Patch(
                facecolor=_SURFACE, edgecolor=_NEUTRAL_COLOR,
                linestyle="--", label="Hidden branches",
            )
        )
    return handles

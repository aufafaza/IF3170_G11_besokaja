from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from ml_lib.model.dtl import DecisionTree, DecisionTreeC45, DecisionTreeCART

TREE_CLASSES = [DecisionTree, DecisionTreeC45, DecisionTreeCART]


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / y.size
    return -np.sum(p * np.log(p))


def gini(y):
    _, counts = np.unique(y, return_counts=True)
    return 1 - np.sum((counts / y.size) ** 2)


def variance(y):
    return np.mean((y - y.mean()) ** 2)


def brute_force_score(mask, y, impurity, ratio=False):
    n_left = mask.sum()
    n_right = y.size - n_left
    gain = impurity(y) - (n_left * impurity(y[mask]) + n_right * impurity(y[~mask])) / y.size
    if not ratio:
        return gain
    p = n_left / y.size
    return gain / -(p * np.log(p) + (1 - p) * np.log(1 - p))


def walk(model, row):
    node = model.root
    while not node.is_leaf_node():
        value = row[node.feature]
        if np.isnan(value):
            node = node.left if node.missing_left else node.right
        elif node.left_categories is not None:
            node = node.left if int(value) in node.left_categories else node.right
        else:
            node = node.left if value <= node.threshold else node.right
    return node.value


@pytest.mark.parametrize(
    ("factory", "impurity", "ratio"),
    [
        (lambda: DecisionTree(max_depth=1), entropy, False),
        (lambda: DecisionTreeC45(max_depth=1), entropy, True),
        (lambda: DecisionTreeCART(max_depth=1), gini, False),
    ],
)
def test_numeric_split_matches_exhaustive_search(factory, impurity, ratio):
    X, y = make_classification(80, 4, n_informative=3, n_redundant=0, random_state=0)
    model = factory().fit(X, y)

    best = max(
        brute_force_score(X[:, f] <= t, y, impurity, ratio)
        for f in range(X.shape[1])
        for t in np.unique(X[:, f])[:-1]
    )
    chosen = X[:, model.root.feature] <= model.root.threshold
    assert brute_force_score(chosen, y, impurity, ratio) == pytest.approx(best)


def test_regression_split_matches_exhaustive_search():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = X[:, 1] ** 2 + rng.normal(scale=0.1, size=80) + 1e6
    model = DecisionTreeCART(regression=True, max_depth=1).fit(X, y)

    best = max(
        brute_force_score(X[:, f] <= t, y, variance)
        for f in range(X.shape[1])
        for t in np.unique(X[:, f])[:-1]
    )
    chosen = X[:, model.root.feature] <= model.root.threshold
    assert brute_force_score(chosen, y, variance) == pytest.approx(best, abs=1e-6)


@pytest.mark.parametrize("regression", [False, True])
def test_categorical_order_finds_best_partition(regression):
    rng = np.random.default_rng(3)
    categories = np.array(list("abcdef"))
    codes = rng.integers(0, categories.size, 200)
    effect = rng.permutation(categories.size) / categories.size
    if regression:
        y = effect[codes] + rng.normal(scale=0.2, size=200)
        impurity = variance
    else:
        y = (rng.random(200) < effect[codes]).astype(int)
        impurity = gini
    X = pd.DataFrame({"c": categories[codes]})

    model = DecisionTreeCART(max_depth=1, regression=regression).fit(X, y)

    best = -np.inf
    for size in range(1, categories.size):
        for subset in combinations(categories, size):
            mask = np.isin(categories[codes], subset)
            best = max(best, brute_force_score(mask, y, impurity))
    chosen_labels = model._encoder.categories_[0][model.root.left_categories]
    chosen = np.isin(categories[codes], list(chosen_labels))
    assert brute_force_score(chosen, y, impurity) == pytest.approx(best)


@pytest.mark.parametrize("tree_class", TREE_CLASSES)
def test_non_alphabetical_category_split_needs_one_level(tree_class):
    X = np.array([["a"], ["b"], ["c"], ["a"], ["b"], ["c"]], dtype=object)
    y = np.array([0, 1, 0, 0, 1, 0])
    model = tree_class(max_depth=1).fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize("tree_class", TREE_CLASSES)
def test_multiclass_categories(tree_class):
    X = pd.DataFrame({"c": list("abcabcabc")})
    y = np.array(["x", "y", "z"] * 3)
    model = tree_class().fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize("tree_class", TREE_CLASSES)
def test_mixed_dataframe_with_nulls(tree_class):
    X = pd.DataFrame({
        "number": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan],
        "category": ["a", None, "b", "a", np.nan, "b"],
        "pandas_category": pd.Categorical(["u", "v", "u", None, "v", "u"]),
    })
    y = np.array([0, 1, 1, 0, 1, 1])
    model = tree_class().fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize("tree_class", TREE_CLASSES)
def test_missing_values_are_not_dropped_and_can_be_learned(tree_class):
    X = np.array([[1.0], [2.0], [3.0], [np.nan], [np.nan], [np.nan]])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = tree_class(max_depth=1).fit(X, y)

    assert model.root.left.n_samples + model.root.right.n_samples == y.size
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_array_equal(model.predict([[np.nan], [99.0]]), [1, 0])


@pytest.mark.parametrize("missing_class", [0, 1])
def test_learned_direction_can_send_missing_left(missing_class):
    X = np.array([[1.0], [2.0], [3.0], [4.0], [np.nan], [np.nan]])
    y = np.array([0, 0, 1, 1, missing_class, missing_class])
    model = DecisionTreeCART(max_depth=1).fit(X, y)

    assert model.root.threshold == pytest.approx(2.5)
    assert model.root.missing_left == (missing_class == 0)
    assert model.predict([[np.nan]])[0] == missing_class


def test_prediction_null_without_training_nulls_uses_larger_child():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([0, 1, 1, 1, 1])
    model = DecisionTreeCART(max_depth=1).fit(X, y)

    assert model.root.left.n_samples == 1
    assert model.predict([[np.nan]])[0] == 1


def test_unseen_category_follows_missing_direction():
    X = pd.DataFrame({"c": ["a", "a", "b", "b", None, None]})
    y = np.array([0, 0, 1, 1, 1, 1])
    model = DecisionTreeCART(max_depth=1).fit(X, y)

    missing = model.predict(pd.DataFrame({"c": [None]}))
    unseen = model.predict(pd.DataFrame({"c": ["new"]}))
    np.testing.assert_array_equal(unseen, missing)


def test_integer_column_can_be_declared_categorical():
    X = pd.DataFrame({"code": [10, 20, 30, 10, 20, 30]})
    y = np.array([0, 1, 0, 0, 1, 0])

    assert DecisionTreeCART(max_depth=1).fit(X, y).score(X, y) < 1
    named = DecisionTreeCART(max_depth=1, categorical_features=["code"]).fit(X, y)
    indexed = DecisionTreeCART(max_depth=1, categorical_features=[0]).fit(X.to_numpy(), y)
    assert named.score(X, y) == 1
    assert indexed.score(X.to_numpy(), y) == 1


def test_null_data_agrees_with_sklearn():
    X, y = make_classification(800, 8, n_informative=5, random_state=7)
    rng = np.random.default_rng(7)
    X[rng.random(X.shape) < 0.1] = np.nan
    train, test = np.arange(600), np.arange(600, 800)

    model = DecisionTreeCART(max_depth=4).fit(X[train], y[train])
    reference = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X[train], y[train])

    agreement = np.mean(model.predict(X[test]) == reference.predict(X[test]))
    assert agreement >= 0.95


def test_pruning_agrees_with_sklearn():
    X, y = make_classification(500, 6, n_informative=4, random_state=11)
    model = DecisionTreeCART(max_depth=6, ccp_alpha=0.01).fit(X, y)
    reference = DecisionTreeClassifier(max_depth=6, ccp_alpha=0.01, random_state=0).fit(X, y)

    assert np.mean(model.predict(X) == reference.predict(X)) >= 0.98


@pytest.mark.parametrize("tree_class", TREE_CLASSES)
def test_batched_predict_matches_node_traversal(tree_class):
    rng = np.random.default_rng(5)
    X = pd.DataFrame({
        "number": rng.normal(size=150),
        "category": rng.choice(["a", "b", "c", None], 150),
    })
    X.loc[rng.random(150) < 0.2, "number"] = np.nan
    y = (X["number"].fillna(0) + (X["category"] == "b") > 0.2).astype(int).to_numpy()
    model = tree_class(max_depth=5).fit(X, y)

    encoded = model._encoder.transform(X)
    expected = [walk(model, row) for row in encoded]
    np.testing.assert_array_equal(model.predict(X), expected)


def test_pruned_tree_predict_uses_compiled_pruned_nodes():
    X, y = make_classification(200, 4, random_state=2)
    model = DecisionTreeCART(max_depth=6, ccp_alpha=0.02).fit(X, y)

    expected = [walk(model, row) for row in X]
    np.testing.assert_array_equal(model.predict(X), expected)


@pytest.mark.parametrize("tree_class", TREE_CLASSES)
def test_random_state_makes_feature_sampling_reproducible(tree_class):
    X, y = make_classification(120, 8, random_state=1)
    first = tree_class(n_features=3, random_state=9).fit(X, y).predict(X)
    second = tree_class(n_features=3, random_state=9).fit(X, y).predict(X)

    np.testing.assert_array_equal(first, second)


def test_global_seed_still_controls_default_sampling():
    X, y = make_classification(120, 8, random_state=1)
    np.random.seed(21)
    first = DecisionTreeCART(n_features=3).fit(X, y).predict(X)
    np.random.seed(21)
    second = DecisionTreeCART(n_features=3).fit(X, y).predict(X)

    np.testing.assert_array_equal(first, second)


def test_clone_keeps_new_parameters():
    model = DecisionTreeCART(categorical_features=["a"], random_state=3, ccp_alpha=0.1)
    copied = clone(model)

    assert copied.get_params() == model.get_params()
    assert copied.root is None


@pytest.mark.parametrize(
    ("X", "y"),
    [
        ([[1], [2]], [0]),
        ([1, 2], [0, 1]),
        ([[1], [2]], [[0, 1], [1, 0]]),
    ],
)
def test_invalid_shapes_raise(X, y):
    with pytest.raises(ValueError):
        DecisionTreeCART().fit(X, y)


def test_predict_rejects_wrong_feature_count():
    model = DecisionTreeCART().fit([[0, 1], [1, 0]], [0, 1])
    with pytest.raises(ValueError, match="features|shape"):
        model.predict([[0]])
    with pytest.raises(ValueError, match="features|shape"):
        model.predict(np.array([[0.0]]))


def test_visualize_shows_categorical_split():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X = pd.DataFrame({"room": ["a", "b", "c", "a", "b", "c"]})
    model = DecisionTreeCART(max_depth=2).fit(X, [0, 1, 0, 0, 1, 0])
    ax = model.visualize(feature_names=["room"])

    assert any(" in {" in text.get_text() for text in ax.texts)
    plt.close("all")

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from ml_lib.model.knn import _BATCH_SIZE, KNN


@pytest.fixture
def data():
    X, y = make_classification(260, 6, n_informative=4, random_state=1)
    return X[:200], y[:200], X[200:], y[200:]


def reference_for(p, k=5):
    if p == np.inf:
        return KNeighborsClassifier(k, metric="chebyshev", algorithm="brute")
    return KNeighborsClassifier(k, p=p, metric="minkowski", algorithm="brute")


def exact_distances(train, query, p):
    difference = np.abs(train - query)
    if p == np.inf:
        return difference.max(axis=1)
    return np.sum(difference ** p, axis=1) ** (1 / p)


def selected_indices(model, X):
    batches = model._neighbor_index_batches(model._prepare_queries(X))
    return np.vstack(list(batches))


def test_default_hyperparameters():
    model = KNN()

    assert model.k == 5
    assert model.p == 2


@pytest.mark.parametrize("p", [1, 1.5, 2, 3, np.inf])
def test_minkowski_neighbors_and_predictions_match_sklearn(data, p):
    X_train, y_train, X_test, _ = data
    model = KNN(k=5, p=p).fit(X_train, y_train)
    reference = reference_for(p).fit(X_train, y_train)

    indices = selected_indices(model, X_test)
    reference_distances, _ = reference.kneighbors(X_test)
    distances = np.array([
        np.sort(exact_distances(X_train[row], query, p))
        for row, query in zip(indices, X_test)
    ])

    np.testing.assert_allclose(distances, reference_distances)
    np.testing.assert_array_equal(model.predict(X_test), reference.predict(X_test))


def test_chebyshev_uses_largest_coordinate():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [5.0, 0.5]])
    model = KNN(k=1, p=np.inf).fit(X, [0, 1, 2])

    # distances from [3, 3] are 3, 1, and 2.5
    assert model.predict([[3.0, 3.0]])[0] == 1


@pytest.mark.parametrize("p", [1, 2, np.inf])
def test_large_feature_offset_keeps_correct_neighbors(data, p):
    X_train, y_train, X_test, _ = data
    offset = 1e9
    model = KNN(k=3, p=p).fit(X_train + offset, y_train)

    indices = selected_indices(model, X_test + offset)
    for row, query in zip(indices, X_test):
        expected = np.sort(exact_distances(X_train, query, p))[:3]
        actual = np.sort(exact_distances(X_train[row], query, p))
        np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.parametrize("p", [1, 2, 3, np.inf])
def test_batching_matches_single_queries(p):
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(80, 4))
    y_train = rng.integers(0, 3, 80)
    X_test = rng.normal(size=(_BATCH_SIZE + 25, 4))
    model = KNN(k=3, p=p).fit(X_train, y_train)

    batched = model.predict(X_test)
    single = np.concatenate([model.predict(X_test[[i]]) for i in range(0, X_test.shape[0], 37)])
    np.testing.assert_array_equal(batched[::37], single)
    assert batched.shape == (X_test.shape[0],)


@pytest.mark.parametrize(
    "labels",
    [np.array([0, 1]), np.array([-1, 1]), np.array(["no", "yes"]), np.array([0.5, 2.5])],
)
def test_label_types_are_preserved(labels):
    X = np.array([[0.0], [0.1], [0.2], [5.0], [5.1], [5.2]])
    y = labels[[0, 0, 0, 1, 1, 1]]
    model = KNN(k=3).fit(X, y)

    predictions = model.predict(X)
    np.testing.assert_array_equal(predictions, y)
    assert predictions.dtype == y.dtype


def test_vote_tie_uses_smallest_label():
    X = np.array([[0.0], [1.0]])
    model = KNN(k=2).fit(X, ["b", "a"])

    assert model.predict([[0.5]])[0] == "a"


def test_multiclass_vote():
    X = np.array([[0.0], [0.1], [5.0], [5.1], [9.0], [9.1]])
    y = np.array([0, 0, 1, 1, 2, 2])

    np.testing.assert_array_equal(KNN(k=2).fit(X, y).predict(X), y)


def test_k_larger_than_training_set_uses_all_rows():
    X = np.array([[0.0], [1.0], [2.0]])
    model = KNN(k=10).fit(X, [1, 1, 0])

    assert model.predict([[2.0]])[0] == 1
    assert next(model.neighbor_labels([[2.0]])).size == 3


def test_neighbor_labels_yields_one_row_per_sample(data):
    X_train, y_train, X_test, _ = data
    model = KNN(k=4).fit(X_train, y_train)

    labels = list(model.neighbor_labels(X_test))
    assert len(labels) == X_test.shape[0]
    assert all(row.shape == (4,) for row in labels)
    votes = np.array([np.sum(row == 1) for row in labels])
    assert ((votes >= 0) & (votes <= 4)).all()


def test_pandas_input(data):
    X_train, y_train, X_test, _ = data
    frame = KNN().fit(pd.DataFrame(X_train), pd.Series(y_train)).predict(pd.DataFrame(X_test))
    array = KNN().fit(X_train, y_train).predict(X_test)

    np.testing.assert_array_equal(frame, array)


def test_empty_query():
    model = KNN(k=1).fit([[0.0], [1.0]], [0, 1])

    assert model.predict(np.empty((0, 1))).shape == (0,)


@pytest.mark.parametrize(
    "params",
    [{"k": 0}, {"k": -1}, {"k": 1.5}, {"p": 0}, {"p": 0.5}, {"p": np.nan}],
)
def test_invalid_hyperparameters_raise(params):
    with pytest.raises(ValueError):
        KNN(**params).fit([[0.0], [1.0]], [0, 1])


def test_hyperparameters_are_checked_after_set_params():
    model = KNN().fit([[0.0], [1.0]], [0, 1]).set_params(k=0)
    with pytest.raises(ValueError, match="k must"):
        model.predict([[0.0]])


@pytest.mark.parametrize(
    ("X", "y"),
    [
        ([0.0, 1.0], [0, 1]),
        ([[0.0], [1.0]], [0]),
        ([[0.0], [1.0]], [[0, 1], [1, 0]]),
        ([[np.nan], [1.0]], [0, 1]),
        ([[np.inf], [1.0]], [0, 1]),
        (np.empty((0, 1)), []),
    ],
)
def test_invalid_training_data_raises(X, y):
    with pytest.raises(ValueError):
        KNN().fit(X, y)


@pytest.mark.parametrize("X", [[[0.0, 1.0]], [0.0], [[np.nan]]])
def test_invalid_query_raises(X):
    model = KNN(k=1).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError):
        model.predict(X)


def test_predict_requires_fit():
    with pytest.raises(RuntimeError, match="must be fitted"):
        KNN().predict([[0.0]])
    with pytest.raises(RuntimeError, match="must be fitted"):
        next(KNN().neighbor_labels([[0.0]]))


def test_clone_and_grid_search(data):
    X_train, y_train, _, _ = data
    model = KNN(k=7, p=np.inf)
    assert clone(model).get_params() == model.get_params()

    search = GridSearchCV(
        KNN(), {"k": [3, 5], "p": [1, 2]}, scoring="accuracy", cv=2, error_score="raise"
    ).fit(X_train, y_train)
    assert np.isfinite(search.best_score_)

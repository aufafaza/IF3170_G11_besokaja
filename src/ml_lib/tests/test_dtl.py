import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.model_selection import GridSearchCV

from ml_lib.model.base import BaseModel
from ml_lib.model.dtl import DecisionTree, DecisionTreeC45


@pytest.fixture(params=[DecisionTree, DecisionTreeC45])
def tree_class(request):
    return request.param


@pytest.mark.parametrize("n_features", [None, 1, 5])
def test_fit_preserves_constructor_parameters(tree_class, n_features):
    X = [[0], [0], [1], [1]]
    y = [0, 0, 1, 1]
    model = tree_class(n_features=n_features)
    params = model.get_params()

    assert isinstance(model, BaseModel)
    assert model.fit(X, y) is model
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_array_equal(model.classes_, [0, 1])
    assert model.n_features_in_ == 1
    assert model.get_params() == params

    copied = clone(model)
    assert copied.get_params() == params
    assert copied.root is None


def test_refit_recomputes_default_feature_count(tree_class):
    model = tree_class()
    model.fit([[0], [1]], [0, 1])
    model.fit([[0, 0], [1, 1]], [0, 1])

    assert model.n_features is None
    assert model._n_features == 2
    np.testing.assert_array_equal(model.predict([[0, 0], [1, 1]]), [0, 1])


def test_predict_requires_fit(tree_class):
    with pytest.raises(RuntimeError, match="must be fitted"):
        tree_class().predict([[0]])


def test_parameter_search(tree_class):
    X = np.array([[0], [1]] * 6)
    y = np.array([0, 1] * 6)
    model = tree_class()
    assert model.set_params(min_samples_split=2) is model
    with pytest.raises(ValueError, match="Invalid parameter"):
        model.set_params(unknown_parameter=1)

    search = GridSearchCV(
        model, {"max_depth": [1, 2]}, scoring="accuracy", cv=2, error_score="raise"
    )
    search.fit(X, y)

    assert search.best_score_ == 1.0
    np.testing.assert_array_equal(search.predict(X), y)


def test_constant_features_predict_majority_label(tree_class):
    X = [[1], [1], [1]]
    model = tree_class().fit(X, [0, 1, 1])

    np.testing.assert_array_equal(model.predict(X), [1, 1, 1])


@pytest.mark.parametrize("labels", [[0.0, 1.0, 1.0], ["no", "yes", "yes"], [-1, 1, 1]])
def test_class_label_types(tree_class, labels):
    X = [[0], [1], [2]]
    model = tree_class().fit(X, labels)

    np.testing.assert_array_equal(model.predict(X), labels)


def test_zero_gain_split_can_learn_xor(tree_class):
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]
    model = tree_class().fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)


def test_classifier_supports_default_parameter_search_scoring(tree_class):
    X = np.array([[0], [1]] * 6)
    y = np.array([0, 1] * 6)
    model = tree_class()
    assert is_classifier(model)

    search = GridSearchCV(model, {"max_depth": [1, 2]}, cv=2, error_score="raise")
    search.fit(X, y)

    assert search.best_score_ == 1.0
    assert search.score(X, y) == 1.0

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, make_blobs, make_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from ml_lib.model.regression import LogisticRegression, elastic_net, l1, l2
from ml_lib.utils.functions import Sigmoid

PENALTIES = [
    lambda: l1(0.3),
    lambda: l2(0.3),
    lambda: elastic_net(0.3, 0.4),
]


@pytest.fixture(scope="module")
def cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return StandardScaler().fit_transform(X), y


def objective(X, y, weights, bias, penalty=None):
    signs = 2.0 * y - 1.0
    loss = np.logaddexp(0.0, -signs * (X @ weights + bias)).mean()
    return loss + (0.0 if penalty is None else penalty(weights))


def test_sigmoid_is_stable_for_extreme_values():
    sigmoid = Sigmoid()
    values = np.array([-1e6, -1000.0, 0.0, 1000.0, 1e6])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        output = sigmoid(values)

    np.testing.assert_allclose(output, [0, 0, 0.5, 1, 1], atol=1e-12)


def test_sigmoid_matches_formula():
    x = np.linspace(-20, 20, 101)
    np.testing.assert_allclose(Sigmoid()(x), 1 / (1 + np.exp(-x)))


@pytest.mark.parametrize("make_penalty", PENALTIES)
def test_penalty_gradient_matches_numeric_gradient(make_penalty):
    penalty = make_penalty()
    w = np.array([0.5, -2.0, 3.0])
    eps = 1e-6
    numeric = np.array([
        (penalty(w + eps * unit) - penalty(w - eps * unit)) / (2 * eps)
        for unit in np.eye(w.size)
    ])

    np.testing.assert_allclose(penalty.grad(w), numeric, atol=1e-6)


@pytest.mark.parametrize("make_penalty", PENALTIES)
def test_prox_minimizes_proximal_objective(make_penalty):
    penalty = make_penalty()
    rng = np.random.default_rng(0)
    v = rng.normal(size=6)
    step = 0.7

    def proximal_objective(u):
        return penalty(u) + np.sum((u - v) ** 2) / (2 * step)

    solution = penalty.prox(v, step)
    best = proximal_objective(solution)
    for _ in range(300):
        candidate = solution + rng.normal(scale=0.05, size=v.size)
        assert proximal_objective(candidate) >= best - 1e-12


def test_prox_closed_forms():
    w = np.array([2.0, -0.05, -3.0])

    np.testing.assert_allclose(l1(1.0).prox(w, 0.1), [1.9, 0.0, -2.9])
    np.testing.assert_allclose(l2(1.0).prox(w, 0.25), w / 1.25)
    np.testing.assert_allclose(
        elastic_net(1.0, 0.5).prox(w, 0.2), np.array([1.9, 0.0, -2.9]) / 1.1
    )


@pytest.mark.parametrize(
    ("make_penalty", "sklearn_params"),
    [
        (lambda alpha: l2(alpha), {"penalty": "l2", "solver": "lbfgs"}),
        (lambda alpha: l1(alpha), {"penalty": "l1", "solver": "saga"}),
        (
            lambda alpha: elastic_net(alpha, 0.5),
            {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.5},
        ),
    ],
)
def test_regularized_objective_matches_sklearn(cancer, make_penalty, sklearn_params):
    X, y = cancer
    alpha = 0.02
    penalty = make_penalty(alpha)
    model = LogisticRegression(learning_rate=0.2, n_iters=6000, penalty=penalty).fit(X, y)
    reference = SklearnLogisticRegression(
        C=1 / (alpha * X.shape[0]), tol=1e-7, max_iter=30000, **sklearn_params
    ).fit(X, y)

    ours = objective(X, y, model.weights, model.bias, penalty)
    theirs = objective(X, y, reference.coef_.ravel(), reference.intercept_[0], penalty)
    assert ours == pytest.approx(theirs, rel=2e-3)
    assert np.mean(model.predict(X) == reference.predict(X)) >= 0.99


def test_l1_and_elastic_net_produce_exact_zero_weights(cancer):
    X, y = cancer
    lasso = LogisticRegression(learning_rate=0.1, n_iters=1500, penalty=l1(0.05)).fit(X, y)
    net = LogisticRegression(
        learning_rate=0.1, n_iters=1500, penalty=elastic_net(0.05, 0.5)
    ).fit(X, y)
    ridge = LogisticRegression(learning_rate=0.1, n_iters=1500, penalty=l2(0.05)).fit(X, y)

    assert (lasso.weights == 0).sum() > 0
    assert (net.weights == 0).sum() > 0
    assert (ridge.weights == 0).sum() == 0


@pytest.mark.parametrize("penalty", [None, l1(0.01), l2(0.01), elastic_net(0.01)])
def test_objective_history_decreases(cancer, penalty):
    X, y = cancer
    model = LogisticRegression(learning_rate=0.1, n_iters=300, penalty=penalty).fit(X, y)

    assert len(model.losses) == 300
    assert np.all(np.diff(model.losses) <= 1e-10)


def test_bias_is_not_penalized():
    X = np.array([[1.0], [-1.0], [1.0], [-1.0], [1.0], [-1.0], [1.0], [1.0]])
    y = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    model = LogisticRegression(learning_rate=0.5, n_iters=2000, penalty=l1(100.0)).fit(X, y)

    np.testing.assert_array_equal(model.weights, [0.0])
    assert model.bias == pytest.approx(np.log(6 / 2), abs=1e-3)


def test_unpenalized_update_matches_formula():
    X = np.array([[1.0, 2.0], [-1.0, 0.5], [0.5, -1.5], [2.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    model = LogisticRegression(learning_rate=0.1, n_iters=1).fit(X, y)

    expected_w = -0.1 * (X.T @ (0.5 - y)) / X.shape[0]
    expected_b = -0.1 * np.mean(0.5 - y)
    np.testing.assert_allclose(model.weights, expected_w)
    assert model.bias == pytest.approx(expected_b)
    assert model.losses[0] == pytest.approx(np.log(2))


def test_large_scores_do_not_warn_or_produce_nan():
    X, y = make_blobs(60, centers=[[-1e4, -1e4], [1e4, 1e4]], random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = LogisticRegression(learning_rate=1.0, n_iters=20).fit(X, y)

    assert np.isfinite(model.losses).all()
    assert np.isfinite(model.weights).all()
    np.testing.assert_array_equal(model.predict(X), y)


@pytest.mark.parametrize(
    "labels",
    [np.array([0, 1]), np.array([-1, 1]), np.array(["no", "yes"]), np.array([2.5, 7.5])],
)
def test_any_two_labels_are_returned(labels):
    X, encoded = make_blobs(80, centers=2, cluster_std=0.8, random_state=3)
    X = StandardScaler().fit_transform(X)
    y = labels[encoded]
    model = LogisticRegression(learning_rate=0.5, n_iters=200).fit(X, y)

    predictions = model.predict(X)
    np.testing.assert_array_equal(predictions, y)
    assert predictions.dtype == y.dtype
    np.testing.assert_array_equal(model.classes_, np.sort(labels))


def test_pandas_indexes_cannot_misalign_training():
    X, y = make_classification(100, 4, random_state=2)
    frame = pd.DataFrame(X)
    target = pd.Series(y, index=np.arange(100)[::-1])

    from_pandas = LogisticRegression(learning_rate=0.1, n_iters=50).fit(frame, target)
    from_numpy = LogisticRegression(learning_rate=0.1, n_iters=50).fit(X, y)

    np.testing.assert_allclose(from_pandas.weights, from_numpy.weights)
    np.testing.assert_array_equal(from_pandas.predict(frame), from_numpy.predict(X))


def test_tol_stops_early_and_default_runs_all_iterations(cancer):
    X, y = cancer
    full = LogisticRegression(learning_rate=0.1, n_iters=2000).fit(X, y)
    early = LogisticRegression(learning_rate=0.1, n_iters=2000, tol=1e-5).fit(X, y)

    assert full.n_iter_ == 2000
    assert len(full.losses) == 2000
    assert early.n_iter_ < 2000
    assert len(early.losses) == early.n_iter_ + 1


def test_custom_penalty_without_prox_uses_gradient():
    class GradientOnlyL2:
        def __call__(self, w):
            return 0.05 * np.dot(w, w)

        def grad(self, w):
            return 0.1 * w

    X, y = make_classification(80, 3, n_redundant=0, random_state=4)
    model = LogisticRegression(learning_rate=0.1, n_iters=100, penalty=GradientOnlyL2()).fit(X, y)

    assert np.isfinite(model.weights).all()


def test_refit_resets_history(cancer):
    X, y = cancer
    model = LogisticRegression(learning_rate=0.1, n_iters=10).fit(X, y)
    model.fit(X, y)

    assert len(model.losses) == 10


def test_feed_forward_returns_probabilities(cancer):
    X, y = cancer
    model = LogisticRegression(learning_rate=0.1, n_iters=50).fit(X, y)
    probabilities = model.feed_forward(X)

    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_array_equal(probabilities > 0.5, model.predict(X) == 1)


@pytest.mark.parametrize(
    "y",
    [
        np.array([0, 1, 2, 0, 1, 2]),
        np.zeros(6, dtype=int),
        np.column_stack([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]]),
    ],
)
def test_non_binary_targets_are_rejected(y):
    X = np.arange(6.0).reshape(-1, 1)
    with pytest.raises(ValueError, match="binary target|exactly 2 classes"):
        LogisticRegression().fit(X, y)


@pytest.mark.parametrize(
    "params",
    [
        {"learning_rate": 0},
        {"learning_rate": -1},
        {"n_iters": 0},
        {"n_iters": 1.5},
        {"tol": -1},
    ],
)
def test_invalid_hyperparameters_raise(params):
    with pytest.raises(ValueError):
        LogisticRegression(**params).fit([[0.0], [1.0]], [0, 1])


@pytest.mark.parametrize("make_penalty", [lambda: l1(-1), lambda: l2(-1), lambda: elastic_net(-1)])
def test_negative_alpha_raises(make_penalty):
    with pytest.raises(ValueError, match="alpha"):
        make_penalty()


@pytest.mark.parametrize("ratio", [-0.1, 1.1])
def test_invalid_l1_ratio_raises(ratio):
    with pytest.raises(ValueError, match="l1_ratio"):
        elastic_net(0.1, ratio)


@pytest.mark.parametrize(
    ("X", "y"),
    [
        ([0.0, 1.0], [0, 1]),
        ([[0.0], [1.0]], [0]),
        ([[np.nan], [1.0]], [0, 1]),
        ([[np.inf], [1.0]], [0, 1]),
    ],
)
def test_invalid_training_data_raises(X, y):
    with pytest.raises(ValueError):
        LogisticRegression().fit(X, y)


@pytest.mark.parametrize("X", [[[0.0, 1.0]], [0.0], [[np.nan]]])
def test_invalid_prediction_data_raises(X):
    model = LogisticRegression(n_iters=1).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError):
        model.predict(X)


def test_predict_requires_fit():
    with pytest.raises(RuntimeError, match="must be fitted"):
        LogisticRegression().predict([[0.0]])


def test_clone_and_grid_search(cancer):
    X, y = cancer
    model = LogisticRegression(learning_rate=0.05, n_iters=20, penalty=l2(0.1), tol=1e-4)
    copied = clone(model)

    assert copied.learning_rate == model.learning_rate
    assert copied.tol == model.tol
    assert isinstance(copied.penalty, l2)
    assert copied.weights is None

    search = GridSearchCV(
        LogisticRegression(n_iters=20),
        {"learning_rate": [0.01, 0.1], "penalty": [None, l1(0.01), l2(0.01), elastic_net(0.01)]},
        scoring="accuracy",
        cv=2,
        error_score="raise",
    ).fit(X, y)
    assert np.isfinite(search.best_score_)

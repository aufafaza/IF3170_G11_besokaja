import warnings

import cvxopt
import matplotlib
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs, make_circles
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml_lib.model.svm import KernelSVM, LinearSVM

matplotlib.use("Agg")

KERNEL_CASES = [
    ("linear", {}, {}),
    ("poly", {"degree": 2, "gamma": 0.5, "coef": 1.0}, {"degree": 2, "gamma": 0.5, "coef0": 1.0}),
    ("rbf", {"gamma": 1.0}, {"gamma": 1.0}),
]


@pytest.fixture
def circles():
    X, y = make_circles(120, factor=0.45, noise=0.08, random_state=2)
    return StandardScaler().fit_transform(X), y


@pytest.fixture
def blobs():
    X, y = make_blobs(100, centers=2, cluster_std=1.1, random_state=4)
    return StandardScaler().fit_transform(X), y


def qp_objective(model, X, y, C):
    signed = np.where(y == model.classes_[1], 1.0, -1.0)
    K = model._kernel_matrix(X, X)
    m = len(y)
    P = cvxopt.matrix(np.outer(signed, signed) * K)
    q = cvxopt.matrix(-np.ones(m))
    G = cvxopt.spmatrix(
        np.r_[-np.ones(m), np.ones(m)],
        np.arange(2 * m),
        np.r_[np.arange(m), np.arange(m)],
        (2 * m, m),
    )
    h = cvxopt.matrix(np.r_[np.zeros(m), np.full(m, float(C))])
    A = cvxopt.matrix(signed, (1, m), "d")
    b = cvxopt.matrix(0.0)
    result = cvxopt.solvers.qp(P, q, G, h, A, b, options={"show_progress": False})
    assert result["status"] == "optimal"
    return result["primal objective"]


@pytest.mark.parametrize(("kernel", "params", "sklearn_params"), KERNEL_CASES)
def test_kernels_match_sklearn_and_qp(circles, kernel, params, sklearn_params):
    X, y = circles
    model = KernelSVM(C=1.0, kernel=kernel, tol=1e-4, **params).fit(X, y)
    reference = SVC(C=1.0, kernel=kernel, tol=1e-4, **sklearn_params).fit(X, y)

    np.testing.assert_allclose(
        model.decision_function(X), reference.decision_function(X), atol=2e-3
    )
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))
    assert model.objective_ == pytest.approx(qp_objective(model, X, y, 1.0), rel=1e-3)


@pytest.mark.parametrize(
    "labels",
    [np.array([0, 1]), np.array([-1, 1]), np.array(["no", "yes"]), np.array([2.5, 7.5])],
)
@pytest.mark.parametrize("model_class", [KernelSVM, LinearSVM])
def test_any_two_labels_are_returned(blobs, model_class, labels, capsys):
    X, encoded = blobs
    y = labels[encoded]
    np.random.seed(0)
    model = model_class().fit(X, y)

    predictions = model.predict(X)
    capsys.readouterr()
    assert set(np.unique(predictions)) <= set(labels)
    assert np.mean(predictions == y) >= 0.95
    np.testing.assert_array_equal(model.classes_, np.sort(labels))


def test_zero_one_labels_do_not_collapse_to_constant_model(blobs):
    X, y = blobs
    model = KernelSVM(gamma=1.0).fit(X, y)

    assert set(model.support_vector_labels) == {-1.0, 1.0}
    assert np.unique(model.predict(X)).size == 2


def test_linear_kernel_is_a_true_dot_product():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    model = KernelSVM(kernel="linear").fit(X, [0, 1])

    np.testing.assert_allclose(model._kernel_matrix(X, X), X @ X.T)


@pytest.mark.parametrize(
    ("alias", "canonical"), [("polynomial", "poly"), ("gaussian", "rbf")]
)
def test_kernel_aliases(blobs, alias, canonical):
    X, y = blobs
    model = KernelSVM(kernel=alias).fit(X, y)

    assert model.kernel_ == canonical
    assert model.get_params()["kernel"] == alias


@pytest.mark.parametrize(
    "params",
    [
        {"kernel": "typo"},
        {"C": 0},
        {"C": -1},
        {"gamma": 0},
        {"degree": 0},
        {"tol": 0},
        {"max_iter": 0},
        {"cache_size": 0},
    ],
)
def test_invalid_hyperparameters_raise(blobs, params):
    X, y = blobs
    with pytest.raises(ValueError):
        KernelSVM(**params).fit(X, y)


@pytest.mark.parametrize("model_class", [KernelSVM, LinearSVM])
def test_multiclass_and_multilabel_are_rejected(model_class):
    X, y = make_blobs(30, centers=3, random_state=0)
    with pytest.raises(ValueError, match="exactly 2 classes"):
        model_class().fit(X, y)

    multilabel = np.column_stack([y == 0, y == 1]).astype(int)
    with pytest.raises(ValueError, match="Multi-label"):
        model_class().fit(X, multilabel)


def test_small_cache_gives_same_solution(circles):
    X, y = circles
    full = KernelSVM(gamma=1.0).fit(X, y)
    small = KernelSVM(gamma=1.0, cache_size=2).fit(X, y)

    np.testing.assert_allclose(small.decision_function(X), full.decision_function(X))


def test_hard_margin_on_separable_data():
    X, y = make_blobs(40, centers=[[-4, -4], [4, 4]], cluster_std=0.5, random_state=0)
    model = KernelSVM(C=None, kernel="linear").fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)
    signed = np.where(y == 1, 1.0, -1.0)
    assert np.min(signed * model.decision_function(X)) >= 1 - 1e-2


def test_max_iter_warns(circles):
    X, y = circles
    with pytest.warns(ConvergenceWarning):
        KernelSVM(max_iter=1).fit(X, y)


def test_ignored_legacy_parameters_warn_and_clone(blobs):
    X, y = blobs
    model = KernelSVM(n_iters=100, learning_rate=0.001)
    copied = clone(model)

    assert copied.get_params() == model.get_params()
    with pytest.warns(FutureWarning, match="ignores n_iters"):
        copied.fit(X, y)


def test_default_kernel_svm_does_not_warn(blobs):
    X, y = blobs
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        KernelSVM().fit(X, y)


def test_predict_requires_fit():
    with pytest.raises(RuntimeError, match="must be fitted"):
        KernelSVM().predict([[0.0, 1.0]])
    with pytest.raises(RuntimeError, match="must be fitted"):
        LinearSVM().predict([[0.0, 1.0]])


def test_calc_cost_uses_fitted_labels(blobs):
    X, y = blobs
    labels = np.array(["no", "yes"])[y]
    model = KernelSVM(gamma=1.0).fit(X, labels)

    weights = model.a * model.support_vector_labels
    K = model._kernel_matrix(model.support_vectors, model.support_vectors)
    signed = np.where(labels == "yes", 1.0, -1.0)
    expected = 0.5 * weights @ K @ weights + np.maximum(
        0, 1 - signed * model.decision_function(X)
    ).sum()
    assert model.calc_cost(X, labels) == pytest.approx(expected)


@pytest.mark.parametrize("C", [0.1, 1, 10])
def test_linear_svm_converges_near_exact_optimum(blobs, C, capsys):
    # Subgradient descent can oscillate slightly, so compare with the true optimum.
    X, y = blobs
    np.random.seed(0)
    model = LinearSVM(C=C, n_iters=500).fit(X, y)
    capsys.readouterr()

    reference = SVC(C=C, kernel="linear", tol=1e-6).fit(X, y)
    w, b = reference.coef_.ravel(), reference.intercept_[0]
    signed = 2.0 * y - 1.0
    optimum = 0.5 * w @ w + C * np.maximum(0, 1 - signed * (X @ w + b)).sum()

    assert list(model.history.values())[-1] == pytest.approx(optimum, rel=1e-3)
    np.testing.assert_array_equal(model.predict(X), reference.predict(X))


def test_linear_svm_rejects_nonlinear_kernel(blobs):
    X, y = blobs
    with pytest.raises(ValueError, match="only supports"):
        LinearSVM(kernel="rbf").fit(X, y)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_linear_svm_plot_handles_fewer_than_fifty_rows(blobs, capsys):
    import matplotlib.pyplot as plt

    X, y = blobs
    np.random.seed(0)
    model = LinearSVM(n_iters=10).fit(X[:20], y[:20])
    model.svm_plot(X[:20], y[:20])
    plt.close("all")
    capsys.readouterr()

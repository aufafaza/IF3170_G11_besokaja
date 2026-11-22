import matplotlib.pyplot as plt
import numpy as np
import pytest

from ml_lib.model.svm import SVM


@pytest.mark.parametrize("C", [0.5, 2.0])
def test_gradient_matches_cost(C):
    X = np.array([[3.0, 2.0], [-1.0, 1.0], [0.5, -0.5]])
    y = np.array([1, -1, 1])
    w = np.array([0.4, -0.1])
    b = 0.2
    model = SVM(C=C)
    gradient_w, gradient_b = model._calc_gradient(X, y, w, b)
    epsilon = 1e-6
    numerical_w = np.empty_like(w)
    for i in range(w.size):
        offset = np.zeros_like(w)
        offset[i] = epsilon
        numerical_w[i] = (
            model.calc_cost(X, y, w + offset, b, C)
            - model.calc_cost(X, y, w - offset, b, C)
        ) / (2 * epsilon)
    numerical_b = (
        model.calc_cost(X, y, w, b + epsilon, C)
        - model.calc_cost(X, y, w, b - epsilon, C)
    ) / (2 * epsilon)

    np.testing.assert_allclose(gradient_w, numerical_w, atol=1e-6)
    np.testing.assert_allclose(gradient_b, numerical_b, atol=1e-6)


def test_history_records_initial_checkpoints_and_final_cost():
    X = np.array([[-1.0], [1.0]])
    y = np.array([-1, 1])
    model = SVM(n_features=1, n_iters=105)
    model.w = np.array([0.2])
    model.b = 0.0
    initial_cost = model.calc_cost(X, y, model.w, model.b, model.C)
    model.fit(X, y)

    assert list(model.history) == [0, 100, 105]
    assert model.history[0] == initial_cost
    assert model.history[105] == pytest.approx(
        model.calc_cost(X, y, model.w, model.b, model.C)
    )
    assert model.history[105] < initial_cost

    model.n_iters = 2
    model.fit(X, y)
    assert list(model.history) == [0, 2]


def test_plot_history_uses_recorded_costs():
    model = SVM(n_features=1, n_iters=2)
    model.fit(np.array([[-1.0], [1.0]]), np.array([-1, 1]))

    ax = model.plot_history(show=False)
    try:
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), list(model.history))
        np.testing.assert_array_equal(
            ax.lines[0].get_ydata(), list(model.history.values())
        )
        assert ax.get_xlabel() == "Iteration"
        assert ax.get_ylabel() == "Cost"
    finally:
        plt.close(ax.figure)


def test_plot_history_requires_fit():
    with pytest.raises(RuntimeError, match="must be fitted"):
        SVM().plot_history(show=False)

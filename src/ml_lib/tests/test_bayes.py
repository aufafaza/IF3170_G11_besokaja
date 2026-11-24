import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB

from ml_lib.model.bayes import DiscreteNB, GaussianNB, MixedNB


@pytest.fixture
def weather():
    X = pd.DataFrame({
        "outlook": ["sun", "sun", "rain", "rain", "cloud", "cloud", "sun", "rain"],
        "windy": ["no", "yes", "no", "yes", "no", "yes", "no", "no"],
    })
    y = np.array(["stay", "stay", "go", "stay", "go", "go", "go", "go"])
    return X, y


def manual_posterior(X, y, row, alpha=1.0):
    classes = np.unique(y)
    joint = []
    for label in classes:
        members = X[y == label]
        probability = len(members) / len(X)
        for column, value in row.items():
            n_values = X[column].nunique(dropna=False)
            count = (members[column] == value).sum()
            probability *= (count + alpha) / (len(members) + alpha * n_values)
        joint.append(probability)
    joint = np.array(joint)
    return classes, joint, joint / joint.sum()


def test_hand_calculated_frequency_smoothing_and_map(weather):
    X, y = weather
    model = DiscreteNB().fit(X, y)

    np.testing.assert_allclose(model.class_prior_, [5 / 8, 3 / 8])
    for outlook in ("sun", "rain", "cloud"):
        for windy in ("no", "yes"):
            row = {"outlook": outlook, "windy": windy}
            classes, joint, posterior = manual_posterior(X, y, row)
            query = pd.DataFrame([row])
            np.testing.assert_allclose(model.predict_proba(query)[0], posterior)
            assert model.predict(query)[0] == classes[np.argmax(joint)]


def test_plus_one_probability_table(weather):
    X, y = weather
    model = DiscreteNB().fit(X, y)

    outlook_categories = list(model._encoder.categories_[0])
    sun = outlook_categories.index("sun")
    # go has 1 sun out of 5 rows, stay has 2 sun out of 3 rows, with 3 outlook values
    np.testing.assert_allclose(
        np.exp(model.feature_log_prob_[0][:, sun]),
        [(1 + 1) / (5 + 3), (2 + 1) / (3 + 3)],
    )
    np.testing.assert_allclose(np.exp(model.feature_log_prob_[0]).sum(axis=1), 1)


def test_matches_sklearn_categorical_nb():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, (300, 5))
    y = (X[:, 0] + rng.integers(0, 2, 300) > 2).astype(int)

    model = DiscreteNB().fit(X, y)
    reference = CategoricalNB(alpha=1.0).fit(X, y)

    np.testing.assert_array_equal(model.predict(X), reference.predict(X))
    np.testing.assert_allclose(model.predict_log_proba(X), reference.predict_log_proba(X))


def test_equal_frequency_bins_are_balanced():
    X = np.arange(1000.0).reshape(-1, 1) + 0.5
    y = (X[:, 0] > 500).astype(int)
    model = DiscreteNB(n_bins=10).fit(X, y)

    assert model.feature_kinds_ == ["binned"]
    assert model.n_values_[0] == 10
    np.testing.assert_array_equal(model.category_count_[0].sum(axis=0), np.full(10, 100))


def test_uniform_bins_have_equal_width():
    rng = np.random.default_rng(1)
    X = np.r_[0.25, rng.uniform(0, 10, 500), 9.75].reshape(-1, 1)
    X[0, 0], X[1, 0] = 0.0, 10.0
    y = (X[:, 0] > 5).astype(int)
    model = DiscreteNB(n_bins=5, strategy="uniform").fit(X, y)

    np.testing.assert_allclose(model.bin_edges_[0], [2, 4, 6, 8])


def test_heavily_tied_column_has_no_empty_bins():
    X = np.r_[np.zeros(90), np.linspace(0.5, 9.5, 10)].reshape(-1, 1)
    y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)]
    model = DiscreteNB(n_bins=5).fit(X, y)

    assert (model.category_count_[0].sum(axis=0) > 0).all()


def test_values_outside_training_range_use_boundary_bins():
    X = np.arange(100.0).reshape(-1, 1) + 0.5
    y = (X[:, 0] > 50).astype(int)
    model = DiscreteNB(n_bins=4).fit(X, y)

    np.testing.assert_array_equal(model.predict([[-1000.0], [1000.0]]), [0, 1])


def test_predict_does_not_change_learned_bins():
    X = np.arange(100.0).reshape(-1, 1) + 0.5
    y = (X[:, 0] > 50).astype(int)
    model = DiscreteNB(n_bins=4).fit(X, y)
    edges = model.bin_edges_[0].copy()

    model.predict(np.linspace(-500, 500, 50).reshape(-1, 1))
    np.testing.assert_array_equal(model.bin_edges_[0], edges)


def test_small_integer_column_keeps_exact_values():
    X = np.array([[0.0], [1.0], [2.0], [0.0], [1.0], [2.0]])
    model = DiscreteNB().fit(X, [0, 1, 0, 0, 1, 0])

    assert model.feature_kinds_ == ["discrete"]
    np.testing.assert_array_equal(model.discrete_values_[0], [0, 1, 2])


def test_unseen_category_uses_count_zero(weather):
    X, y = weather
    model = DiscreteNB().fit(X, y)
    query = pd.DataFrame({"outlook": ["snow"], "windy": ["no"]})

    class_rows = [X[y == label] for label in model.classes_]
    joint = []
    for members in class_rows:
        joint.append(
            len(members) / len(X)
            * (0 + 1) / (len(members) + 3)
            * ((members["windy"] == "no").sum() + 1) / (len(members) + 2)
        )
    joint = np.array(joint)
    np.testing.assert_allclose(model.predict_proba(query)[0], joint / joint.sum())


def test_unseen_discrete_number_uses_count_zero():
    X = np.array([[0.0], [0.0], [1.0], [1.0], [1.0]])
    y = np.array([0, 0, 0, 1, 1])
    model = DiscreteNB().fit(X, y)

    joint = np.array([(3 / 5) * 1 / (3 + 2), (2 / 5) * 1 / (2 + 2)])
    np.testing.assert_allclose(model.predict_proba([[7.0]])[0], joint / joint.sum())


def test_null_is_its_own_attribute_value():
    X = pd.DataFrame({"c": ["a", "a", None, None, "b", "b"]})
    y = np.array([0, 0, 1, 1, 0, 1])
    model = DiscreteNB().fit(X, y)

    assert model.n_values_[0] == 3
    joint = np.array([(3 / 6) * (0 + 1) / (3 + 3), (3 / 6) * (2 + 1) / (3 + 3)])
    posterior = model.predict_proba(pd.DataFrame({"c": [None]}))[0]
    np.testing.assert_allclose(posterior, joint / joint.sum())


def test_unseen_category_is_not_treated_as_null():
    X = pd.DataFrame({"c": ["a", "a", None, None, "b", "b"]})
    y = np.array([0, 0, 1, 1, 0, 1])
    model = DiscreteNB().fit(X, y)

    null = model.predict_proba(pd.DataFrame({"c": [None]}))[0]
    unseen = model.predict_proba(pd.DataFrame({"c": ["new"]}))[0]
    assert not np.allclose(null, unseen)
    np.testing.assert_allclose(unseen, model.class_prior_ * [1 / 6, 1 / 6] / (0.5 / 6 * 2))


def test_numeric_nulls_and_unseen_null_are_supported():
    X = np.array([[1.5], [2.5], [np.nan], [3.5], [np.nan], [4.5]])
    y = np.array([0, 0, 1, 0, 1, 1])
    model = DiscreteNB(n_bins=2).fit(X, y)
    assert model.missing_codes_[0] == model.n_values_[0] - 1
    assert model.predict([[np.nan]])[0] == 1

    no_null_model = DiscreteNB(n_bins=2).fit(X[[0, 1, 3, 5]], y[[0, 1, 3, 5]])
    assert np.isfinite(no_null_model.predict_log_proba([[np.nan]])).all()


def test_all_null_and_constant_columns():
    X = np.column_stack((np.full(6, np.nan), np.ones(6), [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]))
    y = np.array([0, 0, 0, 1, 1, 1])
    model = DiscreteNB(n_bins=2).fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)


def test_log_space_avoids_underflow():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 2, (40, 3000))
    y = np.array([0, 1] * 20)
    model = DiscreteNB().fit(X, y)

    probabilities = model.predict_proba(X)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)


def test_custom_alpha_and_priors(weather):
    X, y = weather
    query = {"outlook": "sun", "windy": "no"}
    model = DiscreteNB(alpha=0.5).fit(X, y)
    _, _, posterior = manual_posterior(X, y, query, alpha=0.5)
    np.testing.assert_allclose(model.predict_proba(pd.DataFrame([query]))[0], posterior)

    weighted = DiscreteNB(priors=[0.9, 0.1]).fit(X, y)
    np.testing.assert_allclose(weighted.class_prior_, [0.9, 0.1])


def test_integer_column_can_be_declared_categorical():
    X = pd.DataFrame({"code": [10, 20, 30, 10, 20, 30] * 3})
    y = np.array([0, 1, 0] * 6)
    model = DiscreteNB(categorical_features=["code"], max_categories=1).fit(X, y)

    assert model.feature_kinds_ == ["categorical"]
    np.testing.assert_array_equal(model.predict(X), y)


def test_clone_and_grid_search(weather):
    X, y = weather
    model = DiscreteNB(alpha=2.0, n_bins=7, strategy="uniform")
    assert clone(model).get_params() == model.get_params()

    X_large = pd.concat([X] * 4, ignore_index=True)
    y_large = np.tile(y, 4)
    search = GridSearchCV(DiscreteNB(), {"alpha": [0.5, 1.0]}, cv=2, error_score="raise")
    search.fit(X_large, y_large)
    assert np.isfinite(search.best_score_)


@pytest.mark.parametrize(
    "params",
    [
        {"alpha": -1},
        {"n_bins": 1},
        {"n_bins": 2.5},
        {"strategy": "bad"},
        {"max_categories": 0},
    ],
)
def test_invalid_hyperparameters_raise(params):
    with pytest.raises(ValueError):
        DiscreteNB(**params).fit([[0], [1]], [0, 1])


@pytest.mark.parametrize(
    ("X", "y"),
    [
        ([[1], [2]], [0]),
        ([1, 2], [0, 1]),
        ([[1], [2]], [[0], [1]]),
        ([[np.inf], [1.5]], [0, 1]),
    ],
)
def test_invalid_input_raises(X, y):
    with pytest.raises(ValueError):
        DiscreteNB().fit(X, y)


def test_predict_requires_fit():
    with pytest.raises(RuntimeError, match="must be fitted"):
        DiscreteNB().predict([[0]])


def test_predict_rejects_wrong_feature_count():
    model = DiscreteNB().fit([[0, 1], [1, 0]], [0, 1])
    with pytest.raises(ValueError):
        model.predict([[0]])


def test_mixed_nb_unseen_value_uses_count_zero():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [0, 0], [1, 1]], float)
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    model = MixedNB(categorical_features=[0, 1]).fit(X, y)

    joint = []
    for label in (0, 1):
        members = X[y == label]
        joint.append(
            len(members) / len(X)
            * 1 / (len(members) + 3)
            * ((members[:, 1] == 0).sum() + 1) / (len(members) + 2)
        )
    joint = np.array(joint)
    np.testing.assert_allclose(model.predict_proba([[9, 0]])[0], joint / joint.sum())


def test_mixed_nb_categorical_matches_sklearn():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, (300, 5)).astype(float)
    y = (X[:, 0] + rng.integers(0, 2, 300) > 2).astype(int)
    model = MixedNB(categorical_features=list(range(5))).fit(X, y)
    reference = CategoricalNB(alpha=1.0).fit(X.astype(int), y)

    np.testing.assert_allclose(
        model.predict_log_proba(X), reference.predict_log_proba(X.astype(int))
    )


def test_mixed_nb_resolves_column_names():
    X = pd.DataFrame({"category": [0, 1, 0, 1], "number": [0.5, 1.5, 2.5, 3.5]})
    model = MixedNB(categorical_features=["category"]).fit(X, [0, 1, 0, 1])

    np.testing.assert_array_equal(model.categorical_features_, [0])
    with pytest.raises(ValueError, match="Unknown columns"):
        MixedNB(categorical_features=["missing"]).fit(X, [0, 1, 0, 1])


def test_gaussian_nb_still_matches_sklearn():
    X, y = load_breast_cancer(return_X_y=True)
    model = GaussianNB().fit(X, y)
    reference = SklearnGaussianNB().fit(X, y)

    np.testing.assert_array_equal(model.predict(X), reference.predict(X))

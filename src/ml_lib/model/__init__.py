"""Re-export the available model classes."""

from .bayes import GaussianNB, MixedNB
from .dtl import DecisionTree
from .knn import KNN
from .regression import LogisticRegression

DTL = DecisionTree

__all__ = [
    "DTL",
    "KNN",
    "DecisionTree",
    "GaussianNB",
    "LogisticRegression",
    "MixedNB",
]

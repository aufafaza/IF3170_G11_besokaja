"""Re-export the available model classes."""

from .dtl import DecisionTree
from .knn import KNN
from .regression import LogisticRegression

DTL = DecisionTree

__all__ = [
    "DTL",
    "DecisionTree",
    "KNN",
    "LogisticRegression",
]

"""Re-export the available model classes."""

from .bayes import DiscreteNB, GaussianNB, MixedNB
from .dtl import DecisionTree, DecisionTreeC45, DecisionTreeCART
from .knn import KNN
from .regression import LogisticRegression, Penalty, elastic_net, l1, l2
from .svm import KernelSVM, LinearSVM

DTL = DecisionTree

__all__ = [
    "DTL",
    "KNN",
    "DecisionTree",
    "DecisionTreeC45",
    "DecisionTreeCART",
    "DiscreteNB",
    "GaussianNB",
    "KernelSVM",
    "LinearSVM",
    "LogisticRegression",
    "MixedNB",
    "Penalty",
    "elastic_net",
    "l1",
    "l2",
]

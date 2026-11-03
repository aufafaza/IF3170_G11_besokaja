"""Re-export class. Make sure the naming is up to date."""

from .bayes_net import BayesianNetwork
from .dtl import DTL
from .knn import KNN
from .regression import LinearRegression, LogisticRegression
from .svm import SVM

__all__ = [
    "DTL",
    "KNN",
    "SVM",
    "BayesianNetwork",
    "LinearRegression",
    "LogisticRegression",
]
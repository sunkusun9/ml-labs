from ._estimator import NNClassifier, NNRegressor
from ._head import SimpleConcatHead
from ._body import DenseBody
from ._tail import LogitTail, BinaryLogitTail, RegressionTail

__all__ = [
    "NNClassifier",
    "NNRegressor",
    "SimpleConcatHead",
    "DenseBody",
    "LogitTail",
    "BinaryLogitTail",
    "RegressionTail",
]

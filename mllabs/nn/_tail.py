from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


class NNTail(BaseEstimator, ABC):
    """Base class for NN tails (output layers).

    Tail receives the Body output tensor and produces the final model output.
    Also owns the loss function and metrics used for model compilation.

    build() contract
    ----------------
    x        : tf.Tensor  from Body
    n_output : int        output dimension (injected by NNClassifier / NNRegressor)
    returns  : tf.Tensor  final output
    """

    @abstractmethod
    def build(self, x, n_output):
        ...

    @abstractmethod
    def loss(self):
        ...

    @abstractmethod
    def compile_metrics(self):
        ...


class LogitTail(NNTail):
    """Multi-class classification output.

    Output: Dense(n_classes, softmax)
    Loss  : sparse_categorical_crossentropy
    """

    def build(self, x, n_output):
        import tensorflow as tf
        return tf.keras.layers.Dense(n_output, activation='softmax', name='logit')(x)

    def loss(self):
        return 'sparse_categorical_crossentropy'

    def compile_metrics(self):
        return ['accuracy']


class BinaryLogitTail(NNTail):
    """Binary classification output.

    Output: Dense(1, sigmoid)
    Loss  : binary_crossentropy
    """

    def build(self, x, n_output):
        import tensorflow as tf
        return tf.keras.layers.Dense(1, activation='sigmoid', name='binary_logit')(x)

    def loss(self):
        return 'binary_crossentropy'

    def compile_metrics(self):
        return ['accuracy']


class RegressionTail(NNTail):
    """Regression output.

    Output: Dense(n_output, linear)
    Loss  : mse
    """

    def build(self, x, n_output):
        import tensorflow as tf
        return tf.keras.layers.Dense(n_output, name='regression')(x)

    def loss(self):
        return 'mse'

    def compile_metrics(self):
        return ['mae']

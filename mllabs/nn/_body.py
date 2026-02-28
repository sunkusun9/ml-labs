from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


class NNBody(BaseEstimator, ABC):
    """Base class for NN bodies.

    Body receives the Head output tensor and applies hidden layers,
    returning a new tensor for Tail to produce the final output.

    build() contract
    ----------------
    x      : tf.Tensor  2D (batch, D) from Head  — or 3D for Transformer variants
    returns: tf.Tensor  same or different shape, passed to Tail
    """

    @abstractmethod
    def build(self, x):
        ...


class DenseBody(NNBody):
    """Standard feed-forward hidden layers.

    Each layer: Dense → [BatchNorm] → Activation → Dropout

    Pairs with: SimpleConcatHead (2D input)
    """

    def __init__(self, layers=(256, 128), dropout=0.3, activation='relu', batch_norm=False):
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm

    def build(self, x):
        import tensorflow as tf

        for i, units in enumerate(self.layers):
            x = tf.keras.layers.Dense(units, name=f'dense_{i}')(x)
            if self.batch_norm:
                x = tf.keras.layers.BatchNormalization(name=f'bn_{i}')(x)
            x = tf.keras.layers.Activation(self.activation, name=f'act_{i}')(x)
            if self.dropout > 0:
                x = tf.keras.layers.Dropout(self.dropout, name=f'drop_{i}')(x)

        return x

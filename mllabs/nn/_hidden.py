try:
    import tensorflow as tf
    _keras_base = tf.keras.Model
except ImportError:
    tf = None
    _keras_base = object


class NNHidden(_keras_base):

    def call(self, x, training=None):
        raise NotImplementedError


class DenseHidden(NNHidden):

    def __init__(self, units=(256, 128), dropout=0.3, activation='relu', batch_norm=False):
        super().__init__()
        self.units = units
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm

        self._blocks = []
        for i, n in enumerate(units):
            self._blocks.append(tf.keras.layers.Dense(n, name=f'dense_{i}'))
            if batch_norm:
                self._blocks.append(tf.keras.layers.BatchNormalization(name=f'bn_{i}'))
            self._blocks.append(tf.keras.layers.Activation(activation, name=f'act_{i}'))
            if dropout > 0:
                self._blocks.append(tf.keras.layers.Dropout(dropout, name=f'drop_{i}'))

    def call(self, x, training=None):
        for layer in self._blocks:
            x = layer(x, training=training)
        return x

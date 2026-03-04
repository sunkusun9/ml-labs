try:
    import tensorflow as tf
    _keras_base = tf.keras.Model
except ImportError:
    tf = None
    _keras_base = object


class NNOutput(_keras_base):

    def set_output_dim(self, n):
        raise NotImplementedError

    def call(self, x, training=None):
        raise NotImplementedError

    def get_loss(self):
        raise NotImplementedError

    def get_metrics(self):
        raise NotImplementedError


class LogitOutput(NNOutput):

    def set_output_dim(self, n):
        self._dense = tf.keras.layers.Dense(n, activation='softmax', name='logit')

    def call(self, x, training=None):
        return self._dense(x, training=training)

    def get_loss(self):
        return 'sparse_categorical_crossentropy'

    def get_metrics(self):
        return ['accuracy']


class BinaryLogitOutput(NNOutput):

    def __init__(self):
        super().__init__()
        self._dense = tf.keras.layers.Dense(1, activation='sigmoid', name='binary_logit')

    def set_output_dim(self, n):
        pass

    def call(self, x, training=None):
        return self._dense(x, training=training)

    def get_loss(self):
        return 'binary_crossentropy'

    def get_metrics(self):
        return ['accuracy']


class RegressionOutput(NNOutput):

    def set_output_dim(self, n):
        self._dense = tf.keras.layers.Dense(n, name='regression')

    def call(self, x, training=None):
        return self._dense(x, training=training)

    def get_loss(self):
        return 'mse'

    def get_metrics(self):
        return ['mae']

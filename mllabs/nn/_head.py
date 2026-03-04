try:
    import tensorflow as tf
    _keras_base = tf.keras.Model
except ImportError:
    tf = None
    _keras_base = object


class NNHead(_keras_base):

    def __init__(self, input_model):
        super().__init__()
        self.input_model = input_model

    def call(self, inputs):
        raise NotImplementedError


class SimpleConcatHead(NNHead):

    def __init__(self, input_model, emb_dropout=0.0):
        super().__init__(input_model)
        self.emb_dropout = emb_dropout

    def call(self, inputs):
        processed = self.input_model(inputs)

        parts = []
        for name in self.input_model._cat_names:
            emb = processed[name]
            if self.emb_dropout > 0:
                emb = tf.keras.layers.Dropout(self.emb_dropout)(emb)
            parts.append(emb)
        for name in self.input_model._cont_names:
            parts.append(processed[name])

        if len(parts) == 1:
            return parts[0]
        return tf.keras.layers.Concatenate()(parts)

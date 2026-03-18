try:
    import tensorflow as tf
    _keras_base = tf.keras.Model
    _layer_base = tf.keras.layers.Layer
except ImportError:
    tf = None
    _keras_base = object
    _layer_base = object

from ._head import NNHead


class FTBlock(_layer_base):

    def __init__(self, d_model, n_heads, ffn_factor, attn_dropout, ffn_dropout, residual_dropout):
        super().__init__()
        self._ln1 = tf.keras.layers.LayerNormalization()
        self._attn = tf.keras.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=d_model // n_heads,
            dropout=attn_dropout,
        )
        self._attn_res_drop = tf.keras.layers.Dropout(residual_dropout) if residual_dropout > 0 else None
        self._ln2 = tf.keras.layers.LayerNormalization()
        ffn_dim = max(1, int(d_model * ffn_factor))
        self._ffn1 = tf.keras.layers.Dense(ffn_dim, activation='gelu')
        self._ffn_drop = tf.keras.layers.Dropout(ffn_dropout) if ffn_dropout > 0 else None
        self._ffn2 = tf.keras.layers.Dense(d_model)
        self._ffn_res_drop = tf.keras.layers.Dropout(residual_dropout) if residual_dropout > 0 else None

    def call(self, x, training=None):
        h = self._ln1(x)
        h = self._attn(h, h, training=training)
        if self._attn_res_drop is not None:
            h = self._attn_res_drop(h, training=training)
        x = x + h

        h = self._ln2(x)
        h = self._ffn1(h)
        if self._ffn_drop is not None:
            h = self._ffn_drop(h, training=training)
        h = self._ffn2(h)
        if self._ffn_res_drop is not None:
            h = self._ffn_res_drop(h, training=training)
        x = x + h
        return x


class FTTransformerHead(NNHead):

    def __init__(
        self,
        input_model,
        d_model=192,
        n_heads=8,
        n_layers=3,
        ffn_factor=4/3,
        attention_dropout=0.2,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    ):
        super().__init__(input_model)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_factor = ffn_factor
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout

        self._cat_projections = {}
        for name, _, ts in input_model._cat_specs:
            emb_dim = ts[1]
            if emb_dim != d_model:
                self._cat_projections[name] = tf.keras.layers.Dense(d_model, use_bias=False)

        n_cont = sum(len(cols) for _, cols, _ in input_model._cont_specs)
        self._n_cont = n_cont
        if n_cont > 0:
            self._cont_w = self.add_weight(
                shape=(n_cont, d_model), initializer='glorot_uniform', name='cont_w'
            )
            self._cont_b = self.add_weight(
                shape=(n_cont, d_model), initializer='zeros', name='cont_b'
            )

        self._cls_token = self.add_weight(
            shape=(1, 1, d_model), initializer='glorot_uniform', name='cls_token'
        )

        self._blocks = [
            FTBlock(d_model, n_heads, ffn_factor, attention_dropout, ffn_dropout, residual_dropout)
            for _ in range(n_layers)
        ]
        self._ln = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=None):
        processed = self.input_model(inputs)
        tokens = []

        for name in self.input_model._cat_names:
            emb = processed[name]
            if name in self._cat_projections:
                emb = self._cat_projections[name](emb)
            tokens.append(tf.expand_dims(emb, 1))

        if self._n_cont > 0:
            cont_parts = [processed[name] for name in self.input_model._cont_names]
            cont = tf.concat(cont_parts, axis=-1) if len(cont_parts) > 1 else cont_parts[0]
            cont_tokens = tf.expand_dims(cont, -1) * self._cont_w + self._cont_b
            tokens.append(cont_tokens)

        all_tokens = tf.concat(tokens, axis=1)

        batch_size = tf.shape(all_tokens)[0]
        cls = tf.tile(self._cls_token, [batch_size, 1, 1])
        x = tf.concat([cls, all_tokens], axis=1)

        for block in self._blocks:
            x = block(x, training=training)

        return self._ln(x[:, 0, :])

from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


class NNHead(BaseEstimator, ABC):
    """Base class for NN heads.

    Head receives per-column embedding inputs/outputs from _EmbeddingEncoder
    and a continuous input tensor, and combines them into a single 2D tensor
    for Body to process.

    build() contract
    ----------------
    emb_inputs  : list[tf.keras.Input]   one per cat col, registered as model inputs
    emb_outputs : list[tf.Tensor]        shape (batch, dim_i) per col
    cont_input  : tf.keras.Input | None  shape (batch, n_cont)
    returns     : tf.Tensor              2D (batch, D) — or 3D for Transformer variants
    """

    @abstractmethod
    def build(self, emb_inputs, emb_outputs, cont_input):
        ...


class SimpleConcatHead(NNHead):
    """Concatenates all embeddings and continuous features into one flat tensor.

    Output shape: (batch, sum(emb_dims) + n_cont)
    Pairs with: DenseBody
    """

    def __init__(self, emb_dropout=0.0):
        self.emb_dropout = emb_dropout

    def build(self, emb_inputs, emb_outputs, cont_input):
        import tensorflow as tf

        parts = []

        for emb in emb_outputs:
            if self.emb_dropout > 0:
                emb = tf.keras.layers.Dropout(self.emb_dropout, name=f'emb_drop')(emb)
            parts.append(emb)

        if cont_input is not None:
            parts.append(cont_input)

        if len(parts) == 1:
            return parts[0]

        return tf.keras.layers.Concatenate()(parts)

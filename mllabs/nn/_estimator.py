import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from ._input import _analyze_cols, _auto_emb_dim, _make_tf_dataset, _DatasetInputModel
from ._head import SimpleConcatHead
from ._body import DenseBody
from ._tail import LogitTail, BinaryLogitTail, RegressionTail

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None


def _iloc(X, indices):
    if pd is not None and isinstance(X, pd.DataFrame):
        return X.iloc[indices]
    if pl is not None and isinstance(X, pl.DataFrame):
        return X[list(indices)]
    if isinstance(X, np.ndarray):
        return X[indices]
    raise TypeError(f"Unsupported data type: {type(X)}")


class _NNBase(BaseEstimator):

    def __init__(
        self,
        cat_cols=None,
        embedding_dims=None,
        head=None,
        body=None,
        tail=None,
        epochs=100,
        batch_size=1024,
        learning_rate=1e-3,
        early_stopping_patience=10,
        validation_fraction=0.1,
        random_state=None,
    ):
        self.cat_cols = cat_cols
        self.embedding_dims = embedding_dims
        self.head = head
        self.body = body
        self.tail = tail
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Column detection
    # ------------------------------------------------------------------

    def _auto_cat_cols(self, X):
        if pd is not None and isinstance(X, pd.DataFrame):
            return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        if pl is not None and isinstance(X, pl.DataFrame):
            return [c for c in X.columns
                    if X[c].dtype == pl.Categorical
                    or (hasattr(pl, 'Enum') and isinstance(X[c].dtype, pl.Enum))]
        raise ValueError("cat_cols must be specified for numpy input")

    def _all_cols(self, X):
        if pd is not None and isinstance(X, pd.DataFrame):
            return list(X.columns)
        if pl is not None and isinstance(X, pl.DataFrame):
            return list(X.columns)
        if isinstance(X, np.ndarray):
            return list(range(X.shape[1]))
        raise TypeError(f"Unsupported data type: {type(X)}")

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _fit_encoder(self, X):
        cat_cols = self._auto_cat_cols(X) if self.cat_cols is None else list(self.cat_cols)
        self.cat_cols_ = cat_cols
        self.cont_cols_ = [c for c in self._all_cols(X) if c not in set(cat_cols)]
        self.col_info_ = _analyze_cols(X, cat_cols) if cat_cols else {}

    def _resolve_embedding_dims(self):
        if not self.cat_cols_:
            return {}
        override = self.embedding_dims or {}
        return {
            col: override.get(col) or _auto_emb_dim(self.col_info_[col]['cardinality'])
            for col in self.cat_cols_
        }

    # ------------------------------------------------------------------
    # var_specs
    # ------------------------------------------------------------------

    def _var_specs(self):
        specs = []
        for col in self.cat_cols_:
            t   = self.col_info_[col]['type']
            ext = 'int' if t in ('ordinal_int', 'int') else 'str'
            dim = self.embedding_dims_[col]
            specs.append((col, [col], ('Embedding', dim, ext)))
        if self.cont_cols_:
            specs.append(('__cont__', self.cont_cols_, 'num'))
        return specs

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    def _make_x_dataset(self, X):
        ds, _ = _make_tf_dataset(X, self._var_specs(),
                                 emb_models=self.input_model_.emb_models)
        return ds

    def _make_ds(self, X, y_encoded):
        ds, _ = _make_tf_dataset(X, self._var_specs(), y=y_encoded,
                                 emb_models=self.input_model_.emb_models)
        return ds

    def _split_val(self, X, y_encoded):
        n = len(y_encoded)
        n_val = max(1, int(n * self.validation_fraction))
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return (
            _iloc(X, train_idx), y_encoded[train_idx],
            _iloc(X, val_idx),   y_encoded[val_idx],
        )

    # ------------------------------------------------------------------
    # Model assembly
    # ------------------------------------------------------------------

    def _build_model(self, X, n_output):
        import tensorflow as tf

        var_specs = self._var_specs()
        _, self.input_model_ = _make_tf_dataset(X, var_specs)

        # Input tensors shaped to match dataset format: (len(cols),) per spec
        inputs_dict = {}
        for name, cols, ts in var_specs:
            if isinstance(ts, tuple) and ts[0] == 'Embedding':
                dtype = 'int32' if ts[2] == 'int' else 'string'
            else:
                dtype = 'float32'
            inputs_dict[name] = tf.keras.Input(shape=(len(cols),), dtype=dtype, name=name)

        processed = self.input_model_(inputs_dict)

        emb_outputs = [processed[col] for col in self.cat_cols_]
        cont_output = processed.get('__cont__')

        head = self.head or SimpleConcatHead()
        x = head.build([], emb_outputs, cont_output)

        body = self.body or DenseBody()
        x = body.build(x)

        self.tail_ = self._resolve_tail()
        output = self.tail_.build(x, n_output)

        model = tf.keras.Model(inputs=list(inputs_dict.values()), outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss=self.tail_.loss(),
            metrics=self.tail_.compile_metrics(),
        )
        return model

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, eval_set=None):
        import tensorflow as tf

        self._fit_encoder(X)
        self.embedding_dims_ = self._resolve_embedding_dims()

        y_encoded = self._prepare_target(y)
        self.model_ = self._build_model(X, self._n_output())

        if eval_set is not None:
            X_val, y_val_raw = eval_set[0]
            y_val = self._encode_y(y_val_raw)
            train_ds = (
                self._make_ds(X, y_encoded)
                .shuffle(len(y_encoded), seed=self.random_state)
                .batch(self.batch_size)
            )
            val_ds = self._make_ds(X_val, y_val).batch(self.batch_size)
        elif self.validation_fraction > 0:
            X_tr, y_tr, X_val, y_val = self._split_val(X, y_encoded)
            train_ds = (
                self._make_ds(X_tr, y_tr)
                .shuffle(len(y_tr), seed=self.random_state)
                .batch(self.batch_size)
            )
            val_ds = self._make_ds(X_val, y_val).batch(self.batch_size)
        else:
            train_ds = (
                self._make_ds(X, y_encoded)
                .shuffle(len(y_encoded), seed=self.random_state)
                .batch(self.batch_size)
            )
            val_ds = None

        callbacks = []
        if self.early_stopping_patience > 0 and val_ds is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                patience=self.early_stopping_patience,
                restore_best_weights=True,
                monitor='val_loss',
            ))

        history = self.model_.fit(
            train_ds,
            epochs=self.epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=0,
        )

        h = history.history
        self.evals_result_ = {}
        train_keys = [k for k in h if not k.startswith('val_')]
        val_keys   = [k for k in h if k.startswith('val_')]
        if train_keys:
            self.evals_result_['train'] = {k: h[k] for k in train_keys}
        if val_keys:
            self.evals_result_['valid'] = {k[4:]: h[k] for k in val_keys}
        self.best_epoch_ = len(h['loss']) - 1

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def _predict_raw(self, X):
        check_is_fitted(self)
        ds = self._make_x_dataset(X).batch(self.batch_size)
        return self.model_.predict(ds, verbose=0)

    # ------------------------------------------------------------------
    # Abstract (subclass)
    # ------------------------------------------------------------------

    def _prepare_target(self, y):
        raise NotImplementedError

    def _encode_y(self, y):
        raise NotImplementedError

    def _n_output(self):
        raise NotImplementedError

    def _resolve_tail(self):
        raise NotImplementedError


# ======================================================================
# NNClassifier
# ======================================================================

class NNClassifier(_NNBase, ClassifierMixin):

    def __init__(
        self,
        cat_cols=None,
        embedding_dims=None,
        head=None,
        body=None,
        tail=None,
        epochs=100,
        batch_size=1024,
        learning_rate=1e-3,
        early_stopping_patience=10,
        validation_fraction=0.1,
        random_state=None,
    ):
        super().__init__(
            cat_cols=cat_cols, embedding_dims=embedding_dims,
            head=head, body=body, tail=tail,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            validation_fraction=validation_fraction, random_state=random_state,
        )

    def _prepare_target(self, y):
        self.classes_ = np.unique(y)
        return self._encode_y(y)

    def _encode_y(self, y):
        return np.searchsorted(self.classes_, np.asarray(y)).astype(np.int32)

    def _n_output(self):
        return len(self.classes_)

    def _resolve_tail(self):
        if self.tail is not None:
            return self.tail
        return BinaryLogitTail() if len(self.classes_) == 2 else LogitTail()

    def predict_proba(self, X):
        raw = self._predict_raw(X)
        if raw.shape[1] == 1:
            return np.hstack([1 - raw, raw])
        return raw

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ======================================================================
# NNRegressor
# ======================================================================

class NNRegressor(_NNBase, RegressorMixin):

    def __init__(
        self,
        cat_cols=None,
        embedding_dims=None,
        head=None,
        body=None,
        tail=None,
        epochs=100,
        batch_size=1024,
        learning_rate=1e-3,
        early_stopping_patience=10,
        validation_fraction=0.1,
        random_state=None,
    ):
        super().__init__(
            cat_cols=cat_cols, embedding_dims=embedding_dims,
            head=head, body=body, tail=tail,
            epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            validation_fraction=validation_fraction, random_state=random_state,
        )

    def _prepare_target(self, y):
        y_arr = np.asarray(y, dtype=np.float32)
        return y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr

    def _encode_y(self, y):
        y_arr = np.asarray(y, dtype=np.float32)
        return y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr

    def _n_output(self):
        return 1

    def _resolve_tail(self):
        return self.tail if self.tail is not None else RegressionTail()

    def predict(self, X):
        return self._predict_raw(X).ravel()

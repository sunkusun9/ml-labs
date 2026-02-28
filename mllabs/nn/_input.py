import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None

try:
    import tensorflow as tf
    _keras_base = tf.keras.Model
except ImportError:
    tf = None
    _keras_base = object


# ======================================================================
# Column analysis
# ======================================================================

def _auto_emb_dim(cardinality):
    return max(1, min(50, (cardinality + 1) // 2))


def _unique_vals(X, col):
    if pd is not None and isinstance(X, pd.DataFrame):
        s = X[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            return s.cat.categories.tolist()
        return s.dropna().unique().tolist()
    if pl is not None and isinstance(X, pl.DataFrame):
        return X[col].drop_nulls().unique().to_list()
    if isinstance(X, np.ndarray):
        arr = X[:, col]
        return [v for v in np.unique(arr) if v is not None]
    raise TypeError(f"Unsupported data type: {type(X)}")


def _is_integer_col(X, col):
    if pd is not None and isinstance(X, pd.DataFrame):
        dtype = X[col].dtype
        if pd.api.types.is_integer_dtype(dtype):
            return True
        if isinstance(dtype, pd.CategoricalDtype):
            cats = dtype.categories
            return cats is not None and len(cats) > 0 and pd.api.types.is_integer_dtype(cats.dtype)
        return False
    if pl is not None and isinstance(X, pl.DataFrame):
        _INT = {pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
        return X[col].dtype in _INT
    if isinstance(X, np.ndarray):
        return np.issubdtype(X[:, col].dtype, np.integer)
    return False


def _is_string_col(X, col):
    if pd is not None and isinstance(X, pd.DataFrame):
        dtype = X[col].dtype
        if dtype == object or isinstance(dtype, pd.StringDtype):
            return True
        if isinstance(dtype, pd.CategoricalDtype):
            cats = dtype.categories
            return cats is not None and len(cats) > 0 and (
                cats.dtype == object or isinstance(cats.dtype, pd.StringDtype)
            )
        return False
    if pl is not None and isinstance(X, pl.DataFrame):
        _STR = {pl.Utf8}
        if hasattr(pl, 'String'):
            _STR.add(pl.String)
        return X[col].dtype in _STR
    if isinstance(X, np.ndarray):
        dtype = X[:, col].dtype
        return np.issubdtype(dtype, np.str_) or dtype == object
    return False


def _is_ordinal(vals):
    try:
        iv = sorted(int(v) for v in vals)
    except (TypeError, ValueError):
        return False
    return bool(iv) and iv[0] == 0 and iv == list(range(len(iv)))


def _analyze_cols(X, col_list):
    col_info = {}
    for col in col_list:
        vals = _unique_vals(X, col)
        cardinality = len(vals)
        if _is_integer_col(X, col) and _is_ordinal(vals):
            col_info[col] = {'type': 'ordinal_int', 'vocab': None, 'cardinality': cardinality}
        elif _is_integer_col(X, col):
            col_info[col] = {'type': 'int', 'vocab': sorted(int(v) for v in vals), 'cardinality': cardinality}
        elif _is_string_col(X, col):
            col_info[col] = {'type': 'string', 'vocab': sorted(str(v) for v in vals), 'cardinality': cardinality}
        else:
            col_info[col] = {'type': 'other', 'vocab': sorted(str(v) for v in vals), 'cardinality': cardinality}
    return col_info


# ======================================================================
# Embedding model builders
# ======================================================================

def build_embedding_models(X, col_list, embedding_dims=None):
    col_info = _analyze_cols(X, col_list)
    models = {}

    for col in col_list:
        info = col_info[col]
        cardinality = info['cardinality']
        dim = (embedding_dims or {}).get(col) or _auto_emb_dim(cardinality)
        t = info['type']

        if t == 'ordinal_int':
            model = tf.keras.Sequential([
                tf.keras.layers.Embedding(cardinality, dim, name=f'emb_{col}'),
            ], name=f'seq_{col}')
        elif t == 'int':
            model = tf.keras.Sequential([
                tf.keras.layers.IntegerLookup(vocabulary=info['vocab'], mask_token=None, name=f'lookup_{col}'),
                tf.keras.layers.Embedding(cardinality + 1, dim, name=f'emb_{col}'),
            ], name=f'seq_{col}')
        elif t == 'string':
            model = tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=info['vocab'], mask_token=None, name=f'lookup_{col}'),
                tf.keras.layers.Embedding(cardinality + 1, dim, name=f'emb_{col}'),
            ], name=f'seq_{col}')
        else:
            model = tf.keras.Sequential([
                tf.keras.layers.Lambda(lambda x: tf.strings.as_string(x), name=f'{col}_to_str'),
                tf.keras.layers.StringLookup(vocabulary=info['vocab'], mask_token=None, name=f'lookup_{col}'),
                tf.keras.layers.Embedding(cardinality + 1, dim, name=f'emb_{col}'),
            ], name=f'seq_{col}')

        models[col] = model

    return models


# ======================================================================
# Data extraction helpers
# ======================================================================

def _dtype_of(type_spec):
    if type_spec == 'num':
        return np.float32
    if type_spec == 'int':
        return np.int32
    if type_spec == 'str':
        return np.object_
    raise ValueError(f"Unknown type_spec: {type_spec!r}")


def _extract(data, cols, dtype):
    cols = list(cols)

    if pd is not None and isinstance(data, pd.DataFrame):
        if dtype == np.object_:
            return np.stack([data[c].astype(str).to_numpy() for c in cols], axis=1)
        if dtype == np.int32:
            arrays = []
            for col in cols:
                s = data[col]
                if isinstance(s.dtype, pd.CategoricalDtype):
                    arrays.append(s.cat.codes.to_numpy().astype(np.int32))
                else:
                    arrays.append(s.to_numpy().astype(np.int32))
            return np.stack(arrays, axis=1)
        return data[cols].to_numpy().astype(dtype)

    if pl is not None and isinstance(data, pl.DataFrame):
        if dtype == np.object_:
            return np.stack([data[c].cast(pl.Utf8).to_numpy() for c in cols], axis=1)
        if dtype == np.int32:
            arrays = []
            for col in cols:
                s = data[col]
                if s.dtype == pl.Categorical or (hasattr(pl, 'Enum') and isinstance(s.dtype, pl.Enum)):
                    arrays.append(s.to_physical().to_numpy().astype(np.int32))
                else:
                    arrays.append(s.to_numpy().astype(np.int32))
            return np.stack(arrays, axis=1)
        return data.select(cols).to_numpy().astype(dtype)

    if isinstance(data, np.ndarray):
        arr = data[:, cols] if data.ndim > 1 else data.reshape(-1, 1)
        return arr.astype(dtype)

    raise TypeError(f"Unsupported data type: {type(data)}")


# ======================================================================
# Input model
# ======================================================================

class _DatasetInputModel(_keras_base):
    """Preprocessing model: categorical columns → embedding, others → passthrough.

    Parameters
    ----------
    cat_specs  : [(name, cols, ('Embedding', dim, ext)), ...]
    emb_models : {name: tf.keras.Sequential}
    cont_specs : [(name, cols, 'num'), ...]
    """

    def __init__(self, cat_specs, emb_models, cont_specs):
        super().__init__()
        self._cat_names = [name for name, _, _ in cat_specs]
        self._cont_names = [name for name, _, _ in cont_specs]
        self.emb_models = emb_models

    def call(self, inputs):
        outputs = {}
        for name in self._cat_names:
            x = tf.squeeze(inputs[name], axis=-1)
            outputs[name] = self.emb_models[name](x)
        for name in self._cont_names:
            outputs[name] = inputs[name]
        return outputs


# ======================================================================
# Dataset factory
# ======================================================================

def _make_tf_dataset(data, var_specs, y=None, emb_models=None):
    """Build a tf.data.Dataset and a _DatasetInputModel from var_specs.

    var_specs format
    ----------------
    categorical : (name, [col], ('Embedding', dim, 'int'|'str'))
    continuous  : (name, cols,  'num')

    Returns
    -------
    (tf.data.Dataset, _DatasetInputModel)
    """
    if tf is None:
        raise ImportError("tensorflow is required")

    cat_specs  = [(n, c, ts) for n, c, ts in var_specs
                  if isinstance(ts, tuple) and ts[0] == 'Embedding']
    cont_specs = [(n, c, ts) for n, c, ts in var_specs
                  if not (isinstance(ts, tuple) and ts[0] == 'Embedding')]

    if cat_specs and emb_models is None:
        col_list       = [cols[0] for _, cols, _ in cat_specs]
        embedding_dims = {name: ts[1] for name, _, ts in cat_specs}
        emb_models     = build_embedding_models(data, col_list, embedding_dims)
    elif emb_models is None:
        emb_models = {}

    tensors = {}
    for name, cols, ts in var_specs:
        if isinstance(ts, tuple) and ts[0] == 'Embedding':
            dtype = np.int32 if ts[2] == 'int' else np.object_
        else:
            dtype = _dtype_of(ts)
        tensors[name] = _extract(data, cols, dtype)

    model = _DatasetInputModel(cat_specs, emb_models, cont_specs)

    if y is None:
        return tf.data.Dataset.from_tensor_slices(tensors), model
    return tf.data.Dataset.from_tensor_slices((tensors, y)), model

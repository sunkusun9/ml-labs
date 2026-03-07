import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import polars as pl
except Exception:
    pl = None

from ._util import detect_kind

_POLARS_TYPES = {
    'str': pl.Utf8 if pl is not None else None,
    'int': pl.Int64 if pl is not None else None,
    'float': pl.Float64 if pl is not None else None,
}


class TypeConverter(BaseEstimator, TransformerMixin):

    def __init__(self, to):
        self.to = to

    def fit(self, X, y=None):
        kind = detect_kind(X)
        if kind == 'numpy':
            self.columns_ = list(range(X.shape[1] if X.ndim > 1 else 1))
        else:
            self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        kind = detect_kind(X)
        if kind == 'pandas_df':
            return X.astype(self.to)
        if kind == 'polars_df':
            dtype = _POLARS_TYPES.get(self.to)
            if dtype is None:
                raise ValueError(f"Unsupported to='{self.to}' for polars. Use 'str', 'int', or 'float'.")
            return X.with_columns([pl.col(c).cast(dtype) for c in X.columns])
        if kind == 'numpy':
            return X.astype(self.to)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return list(input_features)
        if hasattr(self, 'columns_'):
            return [str(c) for c in self.columns_]
        return None

    def set_output(self, transform=None):
        pass

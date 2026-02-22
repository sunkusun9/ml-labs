import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import polars as pl
except Exception:
    pl = None


def detect_kind(X):
    if pd is not None and isinstance(X, pd.DataFrame):
        return "pandas_df"
    if pl is not None and isinstance(X, pl.DataFrame):
        return "polars_df"
    if isinstance(X, np.ndarray):
        return "numpy"
    raise TypeError(f"Unsupported input type: {type(X)}")


def is_nan(x):
    try:
        return isinstance(x, float) and np.isnan(x)
    except Exception:
        return False

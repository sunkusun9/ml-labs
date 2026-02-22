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

from ._util import detect_kind, is_nan


class FrequencyEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, normalize=True):
        self.normalize = normalize

    def fit(self, X, y=None):
        kind = detect_kind(X)
        self.freq_ = {}

        if kind == "pandas_df":
            self.columns_ = list(X.columns)
            for col in self.columns_:
                self.freq_[col] = X[col].value_counts(normalize=self.normalize).to_dict()
        elif kind == "polars_df":
            self.columns_ = list(X.columns)
            for col in self.columns_:
                vc = X.get_column(col).value_counts(normalize=self.normalize)
                prop_col = "proportion" if self.normalize else "count"
                self.freq_[col] = dict(zip(vc.get_column(col).to_list(), vc.get_column(prop_col).to_list()))
        elif kind == "numpy":
            arr = X if X.ndim > 1 else X.reshape(-1, 1)
            self.columns_ = list(range(arr.shape[1]))
            for i in self.columns_:
                unique, counts = np.unique(arr[:, i], return_counts=True)
                vals = (counts / counts.sum()).tolist() if self.normalize else counts.tolist()
                self.freq_[i] = dict(zip(unique.tolist(), vals))

        return self

    def transform(self, X):
        kind = detect_kind(X)
        if kind == "pandas_df":
            return self._transform_pandas(X)
        if kind == "polars_df":
            return self._transform_polars(X)
        if kind == "numpy":
            return self._transform_numpy(X)
        raise TypeError(f"Unsupported input type: {type(X)}")

    def _transform_pandas(self, X):
        return pd.DataFrame(
            {str(col) + '_freq': X[col].map(self.freq_[col]).fillna(0) for col in self.columns_},
            index=X.index
        )

    def _transform_polars(self, X):
        exprs = []
        for col in self.columns_:
            freq_map = self.freq_[col]
            exprs.append(
                pl.col(col).map_elements(
                    lambda x, fm=freq_map: fm.get(x, 0.0),
                    return_dtype=pl.Float64
                ).alias(str(col) + '_freq')
            )
        return X.select(exprs)

    def _transform_numpy(self, X):
        arr = X if X.ndim > 1 else X.reshape(-1, 1)
        result = np.zeros((arr.shape[0], len(self.columns_)), dtype=float)
        for j in range(len(self.columns_)):
            freq_map = self.freq_[self.columns_[j]]
            for i in range(arr.shape[0]):
                result[i, j] = freq_map.get(arr[i, j], 0.0)
        return result

    def get_feature_names_out(self, input_features=None):
        cols = list(input_features) if input_features is not None else [str(c) for c in self.columns_]
        return [col + '_freq' for col in cols]

    def set_output(self, transform=None):
        pass


class CategoricalPairCombiner(BaseEstimator, TransformerMixin):
    """
    두 범주형 변수를 결합해서 하나의 범주형 변수를 만드는 처리기.

    - pairs: [(col_a, col_b), ...]
      * pandas/polars: str(컬럼명) 또는 int(컬럼 위치) 가능
      * numpy: int(컬럼 위치)만 권장
    - min_frequency:
      * fit에서 결합값 빈도를 세고, 빈도가 min_frequency 이하(<=)인 결합값은 transform 시 결측으로 처리

    반환:
      * pandas/polars: 원본 DF에 결합 컬럼을 추가해서 반환
      * numpy: 원본 배열에 결합 컬럼(들)을 뒤에 추가해서 반환 (dtype=object)
    """

    def __init__(
        self,
        pairs,
        min_frequency,
        *,
        sep="__",
        drop_original=False,
        treat_empty_string_as_missing=True,
        missing_value=None,
        new_col_names=None,
    ):
        self.pairs = list(pairs)
        self.min_frequency = int(min_frequency)
        self.sep = sep
        self.drop_original = drop_original
        self.treat_empty_string_as_missing = treat_empty_string_as_missing
        self.missing_value = missing_value
        self.new_col_names = list(new_col_names) if new_col_names is not None else None

        self._allowed_ = {}
        self._categories_ = {}
        self._resolved_pairs_ = []
        self._new_names_ = []

    # ----------------- helpers -----------------

    def _is_missing(self, v):
        if v is None:
            return True
        if is_nan(v):
            return True
        if self.treat_empty_string_as_missing and isinstance(v, str) and v == "":
            return True
        return False

    def _combine(self, a, b):
        if self._is_missing(a) or self._is_missing(b):
            return None
        return f"{a}{self.sep}{b}"

    def _default_new_name(self, a, b):
        return f"{a}{self.sep}{b}"

    def _resolve_pair(self, X, p):
        a, b = p

        # pandas/polars: allow int positional
        if isinstance(X, np.ndarray):
            if not (isinstance(a, int) and isinstance(b, int)):
                raise ValueError("For numpy input, pair elements should be integer column indices.")
            return a, b, str(a), str(b)

        # DataFrame case
        cols = list(X.columns)
        if isinstance(a, int):
            a_name = cols[a]
        else:
            a_name = a
        if isinstance(b, int):
            b_name = cols[b]
        else:
            b_name = b
        return a_name, b_name, str(a_name), str(b_name)

    def _make_new_names(self):
        if self.new_col_names is not None:
            if len(self.new_col_names) != len(self._resolved_pairs_):
                raise ValueError("new_col_names length must match pairs length.")
            return list(self.new_col_names)
        # default names: "colA__colB" (sep 사용)
        names = []
        for _, _, a_label, b_label in self._resolved_pairs_:
            names.append(self._default_new_name(a_label, b_label))
        return names

    # ----------------- sklearn API -----------------

    def fit(self, X, y=None):
        kind = detect_kind(X)
        self._allowed_.clear()
        self._resolved_pairs_.clear()

        # resolve pairs to concrete column selectors/names
        if kind == "numpy":
            for p in self.pairs:
                a_idx, b_idx, a_label, b_label = self._resolve_pair(X, p)
                self._resolved_pairs_.append((a_idx, b_idx, a_label, b_label))
        else:
            for p in self.pairs:
                a_name, b_name, a_label, b_label = self._resolve_pair(X, p)
                self._resolved_pairs_.append((a_name, b_name, a_label, b_label))

        self._new_names_ = self._make_new_names()

        # count frequencies
        for (a_sel, b_sel, a_label, b_label), new_name in zip(self._resolved_pairs_, self._new_names_):
            freq = {}
            if kind == "numpy":
                arr = np.asarray(X)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                col_a = arr[:, a_sel].tolist()
                col_b = arr[:, b_sel].tolist()
                for va, vb in zip(col_a, col_b):
                    comb = self._combine(va, vb)
                    if comb is None:
                        continue
                    freq[comb] = freq.get(comb, 0) + 1
            elif kind == "pandas_df":
                col_a = X[a_sel].tolist()
                col_b = X[b_sel].tolist()
                for va, vb in zip(col_a, col_b):
                    comb = self._combine(va, vb)
                    if comb is None:
                        continue
                    freq[comb] = freq.get(comb, 0) + 1
            elif kind == "polars_df":
                col_a = X.get_column(a_sel).to_list()
                col_b = X.get_column(b_sel).to_list()
                for va, vb in zip(col_a, col_b):
                    comb = self._combine(va, vb)
                    if comb is None:
                        continue
                    freq[comb] = freq.get(comb, 0) + 1
            else:
                raise RuntimeError("Unexpected kind")

            # allowed: 빈도 > min_frequency (<=는 결측 처리)
            allowed = {k for k, c in freq.items() if c > self.min_frequency}
            self._allowed_[new_name] = allowed

            # CategoricalDtype용 categories 구성
            categories = sorted(allowed)
            if self.missing_value is not None and not is_nan(self.missing_value):
                mv = str(self.missing_value)
                if mv not in categories:
                    categories.insert(0, mv)
            self._categories_[new_name] = categories

        return self

    def transform(self, X):
        if not hasattr(self, "_allowed_") or not self._allowed_:
            raise RuntimeError("This transformer is not fitted yet. Call fit() first.")

        kind = detect_kind(X)

        if kind == "pandas_df":
            return self._transform_pandas(X)
        if kind == "polars_df":
            return self._transform_polars(X)
        if kind == "numpy":
            return self._transform_numpy(X)

        raise TypeError(f"Unsupported input type: {type(X)}")

    # ----------------- transforms -----------------

    def _transform_pandas(self, X):
        if pd is None:
            raise RuntimeError("pandas is not installed.")
        X_out = X.copy()

        for (a_name, b_name, _, _), new_name in zip(self._resolved_pairs_, self._new_names_):
            allowed = self._allowed_.get(new_name, set())
            categories = self._categories_.get(new_name)
            dtype = pd.CategoricalDtype(categories=categories) if categories is not None else None

            def _map_row(va, vb, _allowed=allowed):
                comb = self._combine(va, vb)
                if comb is None:
                    return self.missing_value
                if comb in _allowed:
                    return comb
                return self.missing_value

            values = [_map_row(va, vb) for va, vb in zip(X_out[a_name].tolist(), X_out[b_name].tolist())]
            X_out[new_name] = pd.Categorical(values, categories=dtype.categories if dtype else None)

            if self.drop_original:
                X_out = X_out.drop(columns=[a_name, b_name], errors="ignore")

        return X_out

    def _transform_polars(self, X):
        if pl is None:
            raise RuntimeError("polars is not installed.")
        df = X

        exprs = []
        cols_to_drop = []

        for (a_name, b_name, _, _), new_name in zip(self._resolved_pairs_, self._new_names_):
            allowed = self._allowed_.get(new_name, set())

            def _map_struct(s):
                va = s[a_name]
                vb = s[b_name]
                comb = self._combine(va, vb)
                if comb is None:
                    return self.missing_value
                if comb in allowed:
                    return comb
                return self.missing_value

            exprs.append(
                pl.struct([pl.col(a_name), pl.col(b_name)])
                .map_elements(_map_struct, return_dtype=pl.Utf8)
                .cast(pl.Categorical)
                .alias(new_name)
            )

            if self.drop_original:
                cols_to_drop.extend([a_name, b_name])

        if exprs:
            df = df.with_columns(exprs)
        if self.drop_original and cols_to_drop:
            # 중복 제거
            cols_to_drop = list(dict.fromkeys(cols_to_drop))
            df = df.drop(cols_to_drop)

        return df

    def _transform_numpy(self, X):
        arr = np.asarray(X)
        orig_ndim = arr.ndim
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        out = arr.astype(object, copy=True)
        new_cols = []

        for (a_idx, b_idx, _, _), new_name in zip(self._resolved_pairs_, self._new_names_):
            allowed = self._allowed_.get(new_name, set())
            col = np.empty((out.shape[0],), dtype=object)

            for i in range(out.shape[0]):
                comb = self._combine(out[i, a_idx], out[i, b_idx])
                if comb is None:
                    col[i] = self.missing_value
                elif comb in allowed:
                    col[i] = comb
                else:
                    col[i] = self.missing_value

            new_cols.append(col.reshape(-1, 1))

        if new_cols:
            out = np.concatenate([out] + new_cols, axis=1)

        if self.drop_original:
            # 원본 컬럼 제거(쌍에서 나온 컬럼들)
            drop_idxs = sorted({a for (a, _, _, _) in self._resolved_pairs_} | {b for (_, b, _, _) in self._resolved_pairs_})
            keep = [j for j in range(out.shape[1] - len(new_cols)) if j not in drop_idxs]
            # keep 원본 + 새 컬럼들
            out = np.concatenate([out[:, keep]] + new_cols, axis=1)

        if orig_ndim == 1:
            return out
        return out


class CategoricalConverter(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns

    def _resolve_columns(self, X, kind):
        if self.columns is None:
            if kind == "numpy":
                return list(range(X.shape[1]))
            return list(X.columns)
        cols = []
        for c in self.columns:
            if isinstance(c, int) and kind != "numpy":
                cols.append(X.columns[c])
            else:
                cols.append(c)
        return cols

    def fit(self, X, y=None):
        kind = detect_kind(X)
        self.columns_ = self._resolve_columns(X, kind)
        self.kind_ = kind
        return self

    def transform(self, X):
        kind = detect_kind(X)
        if kind == "pandas_df":
            return self._transform_pandas(X)
        if kind == "polars_df":
            return self._transform_polars(X)
        if kind == "numpy":
            return self._transform_numpy(X)
        raise TypeError(f"Unsupported input type: {type(X)}")

    def _transform_pandas(self, X):
        X_out = X.copy()
        for col in self.columns_:
            if col in X_out.columns:
                X_out[col] = X_out[col].astype('category')
        return X_out

    def _transform_polars(self, X):
        exprs = []
        for col in self.columns_:
            if col in X.columns:
                exprs.append(pl.col(col).cast(pl.Categorical))
        if exprs:
            return X.with_columns(exprs)
        return X

    def _transform_numpy(self, X):
        arr = X.copy()
        for col_idx in self.columns_:
            arr[:, col_idx] = arr[:, col_idx].astype(str)
        return arr.astype(object)

    def get_feature_names_out(self, X=None):
        return list(self.columns_) if hasattr(self, 'columns_') else None

    def set_output(self, transform=None):
        pass


class CatOOVFilter(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y = None):
        self.s_dtype_ = {i: X[i].dtype for i in X.columns}
        self.s_mode_ = X.apply(lambda x: x.mode()[0])
        self.fitted_ = True
        return self

    def transform(self, X):
        return pd.concat([
            dproc.rearrange_cat(X[k], v, lambda d, c: 0 if c not in d else c, use_set = True).rename(k)
            for k, v in self.s_dtype_.items()
        ], axis=1)
    def get_params(self, deep=True):
        return {
        }

    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return list(self.s_dtype_.keys())

from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
from ._dproc import get_type_df, get_type_pl, merge_type_df

class PolarsLoader(TransformerMixin, BaseEstimator):
    def __init__(self, predefined_types = {}, read_method = 'read_csv'):
        self.predefined_types = predefined_types
        self.read_method = read_method
    
    def fit(self, X, y = None):
        if type(X) is list:
            self.df_type_ = merge_type_df([
                pl.scan_csv(i).pipe(get_type_df) for i in X
            ])
        else:
            self.df_type_ = pl.scan_csv(X).pipe(get_type_df)
        self.pl_type_ = get_type_pl(
            self.df_type_, self.predefined_types
        )
        self.fitted_ = True
        return self

    def transform(self, X):
        if type(X) is list:
            with pl.StringCache():
                return pl.concat([
                    getattr(pl, self.read_method)(i, schema_overrides=self.pl_type_) for i in X
                ])
        else:
            return getattr(pl, self.read_method)(X, schema_overrides=self.pl_type_)

    def get_params(self, deep = True):
        return {
            'predefined_types': self.predefined_types,
            'read_method': self.read_method
        }
        
    def set_output(self, transform = 'polars'):
        pass

    def get_feature_names_out(self, X = None):
        return [i for i in self.pl_type_.keys()]


class ExprProcessor(TransformerMixin, BaseEstimator):
    def __init__(self, dict_expr, with_columns = True):
        self.dict_expr = dict_expr
        self.with_columns = with_columns
        self.columns = []

    def fit(self, X, y = None):
        if self.with_columns:
            self.columns = X.columns
        self.fitted_ = True
        return self
    
    def transform(self, X):
        if self.with_columns:
            return X.with_columns(**{
                k:v for k, v in self.dict_expr.items()
            })
        else:
            return X.select(**self.dict_expr)
    
    def get_params(self, deep=True):
        return {
            'dict_expr': self.dict_expr,
            'with_columns': self.with_columns
        }

    def set_output(self, transform = 'polars'):
        pass

    def get_feature_names_out(self, X = None):
        return self.columns + list(self.dict_expr.keys())
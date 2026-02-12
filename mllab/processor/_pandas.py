from sklearn.base import BaseEstimator, TransformerMixin

class PandasConverter(TransformerMixin, BaseEstimator):
    def __init__(self, index_col = None):
        self.index_col = index_col
        self.columns_ = None

    def fit(self, X, y = None):
        self.columns_ = [i for i in X.columns if i != self.index_col]
        self.fitted_ = True
        return self
    
    def transform(self, X):
        df = X.to_pandas()
        if self.index_col is not None and self.index_col in df.columns:
            df = df.set_index(self.index_col)
        return df

    def get_params(self, deep = True):
        return {
            'index_col': self.index_col
        }

    def set_output(self, transform = 'pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return self.columns_
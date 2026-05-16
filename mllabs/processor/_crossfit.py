import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold

from .._data_wrapper import DataWrapper


def _head(X, n=2):
    if isinstance(X, np.ndarray):
        return X[:n]
    if hasattr(X, 'iloc'):
        return X.iloc[:n]
    return X[:n]


def _take(arr, indices):
    if isinstance(arr, np.ndarray):
        return arr[indices]
    if hasattr(arr, 'iloc'):
        return arr.iloc[indices]
    return arr[indices]


class CrossFitTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, estimator, cv=5, method='predict_proba', stratified=True):
        self.estimator = estimator
        self.cv = cv
        self.method = method
        self.stratified = stratified

    def _make_cv(self, y):
        if not isinstance(self.cv, int):
            return self.cv
        if self.stratified and y is not None:
            return StratifiedKFold(n_splits=self.cv)
        return KFold(n_splits=self.cv)

    def _predict(self, estimator, X):
        pred = getattr(estimator, self.method)(X)
        if pred.ndim == 1:
            return pred.reshape(-1, 1)
        return pred

    def _to_input_type(self, pred, X):
        wrapper = DataWrapper.from_native(X)
        names = self.get_feature_names_out()
        index = wrapper.get_index()
        return type(wrapper).from_output(pred, names, index).to_native()

    def fit(self, X, y=None):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        sample = self._predict(self.estimator_, _head(X))
        self.n_outputs_ = sample.shape[1]
        if hasattr(self.estimator_, 'classes_'):
            self.classes_ = self.estimator_.classes_
        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)

        cv = self._make_cv(y)
        n = len(X)
        oof = np.zeros((n, self.n_outputs_))

        for train_idx, val_idx in cv.split(X, y):
            est = clone(self.estimator)
            est.fit(_take(X, train_idx), _take(y, train_idx))
            oof[val_idx] = self._predict(est, _take(X, val_idx))

        return self._to_input_type(oof, X)

    def transform(self, X):
        return self._to_input_type(self._predict(self.estimator_, X), X)

    def get_feature_names_out(self, input_features=None):
        prefix = type(self.estimator).__name__.lower()
        if self.method == 'predict':
            return [f'{prefix}_pred']
        if hasattr(self, 'classes_'):
            return [f'{prefix}_{c}' for c in self.classes_]
        return [f'{prefix}_{i}' for i in range(self.n_outputs_)]

    def set_output(self, transform=None):
        pass


from sklearn.base import BaseEstimator, TransformerMixin

class LGBMIterativeImputer(TransformerMixin, BaseEstimator):
    def __init__(self, hparams, X_num, X_cat, targets, na_value = None, hparams_dic = {}, na_value_dic = {}, 
                 n_iter = 2, validation_fraction = 0, progress_callback=None):
        self.hparams = hparams
        self.X_num = X_num
        self.X_cat = X_cat
        self.targets = targets
        self.na_value = na_value
        self.hparams_dic = hparams_dic
        self.na_value_dic = na_value_dic
        self.n_iter = n_iter
        self.validation_fraction = validation_fraction
        self.models_ = None
        self.hist_ = None
        self.initial_cat_values_ = None
        self.progress_callback = progress_callback

    def fit(self, X, y = None):
        self.models_ = list()
        self.hist_ = list()
        self.progress_callback.on_train_begin()
        X_ = X.copy()
        if len(self.X_cat) > 0:
            self.initial_cat_values_ = X_[self.X_cat].apply(lambda x: x.mode().iloc[0])
            X_ = dproc.join_and_assign(
                X_, X_[self.X_cat].fillna(self.initial_cat_values_)
            )
        for i in range(self.n_iter):
            self.progress_callback.on_iter_begin(self.n_iter)
            self._partial_fit(X_, X)
            self.progress_callback.on_iter_end(i)
        self.progress_callback.on_train_end()
        del X_
        self.fitted_ = True
        return self

    def _partial_fit(self, X, X_org):
        if self.models_ is None:
            self.models_ = list()
            self.hist_ = list()
        if len(self.models_) >= self.n_iter:
            return self
        round_1 = list()
        round_hist = list()
        self.progress_callback.on_step_begin(len(self.targets))
        for target in self.targets:
            val = [i for i in self.X_num + self.X_cat if i != target]
            hparams = self.hparams_dic.get(target, self.hparams)
            na_value = self.na_value_dic.get(target, self.na_value)
            s_tgt_notna = X_org[target].notna() if na_value is None else X_org[target] != na_value
            if target in self.X_cat:
                round_1.append(
                    lgb.LGBMClassifier(verbose = - 1, **hparams)
                )
                if self.validation_fraction <= 0:
                    X_train = X.loc[s_tgt_notna]
                    X_test = None
                else:
                    X_train, X_test = X.loc[s_tgt_notna].pipe(
                        lambda x: train_test_split(x, test_size = self.validation_fraction, random_state = 123, stratify = x[target])
                    )
            else:
                round_1.append(
                    lgb.LGBMRegressor(verbose = -1, **hparams)
                )
                if self.validation_fraction <= 0:
                    X_train = X.loc[s_tgt_notna]
                    X_test = None
                else:
                    X_train, X_test = X.loc[s_tgt_notna].pipe(
                        lambda x: train_test_split(x, test_size = self.validation_fraction, random_state = 123)
                    )
            round_1[-1].fit(X_train[val], X_train[target], categorical_feature = [i for i in val if i in self.X_cat])
            X.loc[~s_tgt_notna, target] = X.loc[~s_tgt_notna, val].pipe(lambda x: pd.Series(round_1[-1].predict(x), index = x.index, dtype = X[target].dtype))
            if X_test is not None:
                if target in self.X_cat:
                    round_hist.append(
                        accuracy_score(X_test[target], round_1[-1].predict(X_test[val]))
                    )
                else:
                    round_hist.append(
                        mean_squared_error(X_test[target], round_1[-1].predict(X_test[val]))
                    )
            self.progress_callback.on_step_end(len(self.targets))
        self.models_.append(round_1)
        self.hist_.append(round_hist)
        return self
    
    def transform(self, X, n_iter = None):
        if n_iter is None:
            n_iter = len(self.models_)
        X_ = X.copy()
        if len(self.X_cat) > 0:
            self.initial_cat_values_ = X_[self.X_cat].apply(lambda x: x.mode().iloc[0])
            X_ = dproc.join_and_assign(
                X_, X_[self.X_cat].fillna(self.initial_cat_values_)
            )
        for i, round_1 in enumerate(self.models_):
            if i >= n_iter:
                break
            for target, m in zip(self.targets, round_1):
                na_value = self.na_value_dic.get(target, self.na_value)
                s_tgt_na = X[target].isna() if na_value is None else X[target] == na_value
                na_value = self.na_value_dic.get(target, self.na_value)
                val = [i for i in self.X_num + self.X_cat if i != target]
                X_.loc[s_tgt_na, target] = X_.loc[s_tgt_na, val].pipe(lambda x: pd.Series(m.predict(x), index = x.index, dtype = X[target].dtype))
        return X_[self.targets]

    def get_params(self, deep=True):
        return {
            'hparams':  hparams,
            'X_num': X_num,
            'X_cat': X_cat,
            'targets': targets,
            'na_value': na_value,
            'hparams_dic': hparams_dic,
            'na_value_dic': na_value_dic,
            'n_iter': n_iter,
            'validate_fraction': validate_fraction
        }
    
    def set_output(self, transform='pandas'):
        pass

    def get_feature_names_out(self, X = None):
        return [target]
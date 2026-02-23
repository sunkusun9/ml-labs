"""
XGBoost adapter
"""

import pandas as pd
from ._base import ModelAdapter

from xgboost.callback import TrainingCallback

class ProgressCallback(TrainingCallback):
    def __init__(self, n_estimators, period_pct, logger):
        self.n_estimators = n_estimators
        self.period_pct = period_pct
        self.last_printed = -1
        self.logger = logger

    def after_iteration(self, model, epoch, evals_log):
        current = epoch + 1
        percentage = (current / self.n_estimators) * 100

        # period_pct마다 출력
        if int(percentage / (self.period_pct * 100)) > self.last_printed:
            self.last_printed = int(percentage / (self.period_pct * 100))

            # metric 정보 추출
            metrics_str = ""
            if evals_log:
                last_metrics = []
                for dataset, metrics in evals_log.items():
                    for metric_name, values in metrics.items():
                        last_metrics.append(f"{dataset}-{metric_name}: {values[-1]:.4f}")
                metrics_str = ", ".join(last_metrics)
            self.logger.adhoc_progress(current, self.n_estimators, metrics_str if metrics_str else None)
        return False

class XGBoostAdapter(ModelAdapter):
    """Adapter for XGBoost models (XGBClassifier, XGBRegressor)

    XGBoost는 eval_set 파라미터로 [(X, y), ...] 형태를 받습니다.
    """

    result_objs = [
        'feature_importances_weight', 'feature_importances_gain', 'feature_importances_cover',
        'feature_importances_total_gain', 'feature_importances_total_cover',
        'evals_result', 'trees'
    ]

    def get_params(self, params, logger = None):
        """XGBoost 모델 생성자 파라미터 조정 (ProgressCallback 설정)"""
        if params is None:
            params = {}
        else:
            params = params.copy()
        if self.verbose > 0 and self.verbose < 1:
            # 0 < verbose < 1: 진행률 기반 출력을 위한 callback 설정

            n_estimators = params.get('n_estimators', 100)
            callbacks = params.get('callbacks', [])
            if logger is not None:
                callbacks.append(ProgressCallback(n_estimators, self.verbose, logger))
            params['callbacks'] = callbacks

        return params

    def get_fit_params(self, data_dict, params=None, logger=None):
        """XGBoost의 fit 파라미터 구성"""
        from .._data_wrapper import unwrap

        fit_params = {}
        if params is not None and params.get('verbosity', 0) > 0:
            fit_params['verbose'] = True
        else:
            fit_params['verbose'] = False

        # data_dict에서 데이터 추출
        train_X, train_v_X = data_dict['X']
        if 'y' in data_dict:
            train_y, train_v_y = data_dict['y']
        else:
            train_y, train_v_y = None, None

        # eval_set 구성
        if self.eval_mode and self.eval_mode != 'none' and train_v_X is not None and train_v_y is not None:
            train_X_native = unwrap(train_X)
            train_y_native = unwrap(train_y)
            train_v_X_native = unwrap(train_v_X)
            train_v_y_native = unwrap(train_v_y)

            if self.eval_mode == 'valid':
                fit_params['eval_set'] = [(train_v_X_native, train_v_y_native)]
            elif self.eval_mode == 'both':
                fit_params['eval_set'] = [(train_X_native, train_y_native), (train_v_X_native, train_v_y_native)]

        return fit_params

    def _get_feature_importances(processor, importance_type):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.n_features_in_))
        booster = obj.get_booster()
        scores = booster.get_score(importance_type=importance_type)

        return pd.Series(
            [scores.get(f, 0) for f in input_vars],
            index=input_vars, name = 'importance'
        )

    def _get_evals_result(processor):
        obj = processor.obj
        evals_result =  obj.evals_result() if hasattr(obj, 'evals_result') else {}
        return pd.concat(
            [pd.DataFrame(v).stack().rename(k) for k, v in evals_result.items()], axis=1
        ).stack()

    def _get_trees(processor):
        obj = processor.obj
        booster = obj.get_booster()
        return booster.trees_to_dataframe()


XGBoostAdapter.result_objs = {
    'feature_importances': (XGBoostAdapter._get_feature_importances, True),
    'evals_result': (XGBoostAdapter._get_evals_result, True), 
    'trees': (XGBoostAdapter._get_trees, False)
}
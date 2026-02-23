"""
LightGBM adapter
"""

import pandas as pd
from ._base import ModelAdapter
from lightgbm import early_stopping

def create_progress_callback(n_estimators, period_pct, logger):
    last_printed = [-1]  # mutable object
    def callback(env):
        current = env.iteration + 1
        percentage = (current / n_estimators) * 100

        # period_pct마다 출력
        if int(percentage / (period_pct * 100)) > last_printed[0]:
            last_printed[0] = int(percentage / (period_pct * 100))

            # metric 정보 추출
            metrics_str = ""
            if env.evaluation_result_list:
                last_metrics = []
                for item in env.evaluation_result_list:
                    dataset_name, metric_name, value, _ = item
                    last_metrics.append(f"{dataset_name}-{metric_name}: {value:.4f}")
                metrics_str = ", ".join(last_metrics)
            logger.adhoc_progress(current, n_estimators, metrics_str if metrics_str else None)

    return callback

class LightGBMAdapter(ModelAdapter):
    """Adapter for LightGBM models (LGBMClassifier, LGBMRegressor)

    LightGBM도 eval_set 파라미터를 사용하지만 약간 다른 방식입니다.
    """
    def get_params(self, params, logger = None):
        """모델 생성자에 전달할 파라미터를 조정

        Args:
            params (dict): 원본 파라미터

        Returns:
            dict: 조정된 파라미터
        """
        return {k: v for k, v in params.items() if k not in ['early_stopping', 'eval_metric']}

    def get_fit_params(self, data_dict, params=None, logger=None):
        """LightGBM의 fit 파라미터 구성"""
        from .._data_wrapper import unwrap

        fit_params = {}

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

        # verbose 처리
        if self.verbose > 0:
            if self.verbose < 1:
                # 0 < verbose < 1: 진행률 기반 출력
                # n_estimators 추출 (기본값 100)
                n_estimators = params.get('n_estimators', 100) if params else 100
                callbacks = fit_params.get('callbacks', [])
                if logger is not None:
                    callbacks.append(create_progress_callback(n_estimators, self.verbose, logger))
                fit_params['callbacks'] = callbacks
            else:
                # verbose >= 1: LightGBM 기본 verbose (iteration 단위)
                fit_params['verbose'] = int(self.verbose)
        
        if 'early_stopping' in params:
            if 'callbacks' not in fit_params:
                fit_params['callbacks'] = list()
            fit_params['callbacks'].append(params['early_stopping'])
        if 'eval_metric' in params:
            fit_params['eval_metric'] = params['eval_metric']
        return fit_params

    @staticmethod
    def _get_feature_importances(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.n_features_in_))

        return pd.Series(
            obj.feature_importances_,
            index=input_vars, name = 'importance'
        )

    @staticmethod
    def _get_evals_result(processor):
        obj = processor.obj
        evals_result = obj.evals_result_ if hasattr(obj, 'evals_result_') else {}
        return pd.concat(
            [pd.DataFrame(v).stack().rename(k) for k, v in evals_result.items()], axis=1
        ).stack()

    @staticmethod
    def _get_trees(processor):
        obj = processor.obj
        dump = obj.booster_.dump_model()
        return dump.get('tree_info', [])

LightGBMAdapter.result_objs = {
    'feature_importances': (LightGBMAdapter._get_feature_importances, True),
    'evals_result': (LightGBMAdapter._get_evals_result, True),
    'trees': (LightGBMAdapter._get_trees, False)
}
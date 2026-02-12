"""
CatBoost adapter
"""

import tempfile
import json
import pandas as pd
from ._base import ModelAdapter


class CatBoostAdapter(ModelAdapter):
    """Adapter for CatBoost models (CatBoostClassifier, CatBoostRegressor)

    CatBoost도 eval_set을 지원합니다.
    """

    def get_fit_params(self, data_dict, X, y=None, params=None, logger=None):
        """CatBoost의 fit 파라미터 구성"""
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
                # CatBoost는 복잡한 callback 구조라서 간단히 기본 verbose 사용
                fit_params['verbose'] = False
            else:
                # verbose >= 1: CatBoost 기본 verbose (iteration 단위)
                fit_params['verbose'] = int(self.verbose)
        else:
            # verbose == 0: 출력 안함
            fit_params['verbose'] = False

        return fit_params

    @staticmethod
    def _get_feature_importances_pvc(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.feature_count_))

        return pd.Series(
            obj.get_feature_importance(type='PredictionValuesChange'),
            index=input_vars, name = 'PredictionValuesChange'
        )

    @staticmethod
    def _get_feature_importances_interaction(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.feature_count_))

        interaction = obj.get_feature_importance(type='Interaction')
        return pd.DataFrame(
            interaction, columns=['feat1', 'feat2', 'importance']
        ).assign(
            feat1=lambda x: x['feat1'].astype('int').apply(lambda y: input_vars[y]),
            feat2=lambda x: x['feat2'].astype('int').apply(lambda y: input_vars[y]),
        ).set_index(['feat1', 'feat2'])['importance']

    @staticmethod
    def _get_evals_result(processor):
        obj = processor.obj
        evals_result = obj.get_evals_result() if hasattr(obj, 'get_evals_result') else {}
        return pd.concat(
            [pd.DataFrame(v).stack().rename(k) for k, v in evals_result.items()], axis=1
        ).stack()

    @staticmethod
    def _get_trees(processor):
        obj = processor.obj
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            obj.save_model(f.name, format="json")
            trees = json.load(f).get('oblivious_trees', [])
        return trees

CatBoostAdapter.result_objs = {
    'feature_importances_pvc': (CatBoostAdapter._get_feature_importances_pvc, True),
    'feature_importances_interaction': (CatBoostAdapter._get_feature_importances_interaction, True),
    'evals_result': (CatBoostAdapter._get_evals_result, True), 
    'trees': (CatBoostAdapter._get_trees, False)
}
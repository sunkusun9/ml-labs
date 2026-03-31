"""
Default adapter for sklearn-compatible models
"""

from ._base import ModelAdapter


class DefaultAdapter(ModelAdapter):
    """Default adapter for models that don't support eval_set

    일반적인 sklearn 모델들을 위한 기본 어댑터
    eval_set을 지원하지 않으므로 일반 fit()만 수행
    """

    def get_fit_params(self, train_data, valid_data=None, params=None, monitor=None):
        return super().get_fit_params(train_data, valid_data, params, monitor)

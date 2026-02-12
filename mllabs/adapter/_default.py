"""
Default adapter for sklearn-compatible models
"""

from ._base import ModelAdapter


class DefaultAdapter(ModelAdapter):
    """Default adapter for models that don't support eval_set

    일반적인 sklearn 모델들을 위한 기본 어댑터
    eval_set을 지원하지 않으므로 일반 fit()만 수행
    """

    def get_fit_params(self, data_dict, X, y=None, params=None, logger=None):
        """기본 fit 파라미터 (추가 파라미터 없음)

        eval_set 관련 파라미터는 무시합니다.
        """
        # eval_set을 지원하지 않으므로 빈 dict 반환
        return {}

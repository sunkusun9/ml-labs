"""
Keras adapter
"""

from ._base import ModelAdapter, GPU_POSSIBLE


class KerasAdapter(ModelAdapter):
    """Adapter for Keras models (KerasClassifier, KerasRegressor)

    Keras는 validation_data 파라미터로 (X, y) 튜플을 받습니다.
    """

    def get_gpu_usage(self, params):
        gpu = (params or {}).get('gpu', 'auto')
        if gpu != 'auto':
            raise ValueError(f"KerasAdapter only supports gpu='auto', got {gpu!r}")
        return GPU_POSSIBLE

    def get_params(self, params, monitor=None):
        if params is None:
            return {}
        params = params.copy()
        gpu = params.pop('gpu', 'auto')
        if gpu != 'auto':
            raise ValueError(f"KerasAdapter only supports gpu='auto', got {gpu!r}")
        return params

    def get_fit_params(self, train_data, valid_data=None, params=None, monitor=None):
        """Keras의 fit 파라미터 구성"""
        from .._data_wrapper import unwrap

        fit_params = super().get_fit_params(train_data, valid_data, params, monitor)

        train_v_X = valid_data.get('X') if valid_data else None
        train_v_y = valid_data.get('y') if valid_data else None

        if self.eval_mode and self.eval_mode != 'none' and train_v_X is not None and train_v_y is not None:
            fit_params['validation_data'] = (unwrap(train_v_X), unwrap(train_v_y))

        if self.verbose > 0:
            if self.verbose < 1:
                fit_params['verbose'] = 1
            else:
                fit_params['verbose'] = min(int(self.verbose), 2)
        else:
            fit_params['verbose'] = 0

        return fit_params

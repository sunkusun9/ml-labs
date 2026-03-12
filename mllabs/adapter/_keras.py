"""
Keras adapter
"""

from ._base import ModelAdapter


class KerasAdapter(ModelAdapter):
    """Adapter for Keras models (KerasClassifier, KerasRegressor)

    Keras는 validation_data 파라미터로 (X, y) 튜플을 받습니다.
    """

    def get_fit_params(self, data_dict, params=None, logger=None):
        """Keras의 fit 파라미터 구성"""
        from .._data_wrapper import unwrap

        fit_params = super().get_fit_params(data_dict, params, logger)

        train_v_X = data_dict['X'][1]
        train_v_y = data_dict['y'][1] if 'y' in data_dict else None

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

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

        fit_params = {}

        # data_dict에서 데이터 추출
        train_X, train_v_X = data_dict['X']
        if 'y' in data_dict:
            train_y, train_v_y = data_dict['y']
        else:
            train_y, train_v_y = None, None

        # validation_data 구성
        if self.eval_mode and self.eval_mode != 'none' and train_v_X is not None and train_v_y is not None:
            train_v_X_native = unwrap(train_v_X)
            train_v_y_native = unwrap(train_v_y)
            # Keras는 'valid'와 'both' 모두 동일하게 처리 (validation_data만 지원)
            fit_params['validation_data'] = (train_v_X_native, train_v_y_native)

        # verbose 처리
        if self.verbose > 0:
            if self.verbose < 1:
                # 0 < verbose < 1: Keras의 verbose는 0, 1, 2만 지원하므로 1 사용
                fit_params['verbose'] = 1
            else:
                # verbose >= 1: Keras verbose (0: silent, 1: progress bar, 2: one line per epoch)
                fit_params['verbose'] = min(int(self.verbose), 2)
        else:
            # verbose == 0: 출력 안함
            fit_params['verbose'] = 0

        return fit_params

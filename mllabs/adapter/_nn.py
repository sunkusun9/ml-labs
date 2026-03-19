import pandas as pd
from ._base import ModelAdapter, GPU_NO, GPU_POSSIBLE, GPU_YES

try:
    import tensorflow as tf
    _keras_cb = tf.keras.callbacks.Callback
except ImportError:
    tf = None
    _keras_cb = object


class _ProgressCallback(_keras_cb):

    def __init__(self, n_epochs, verbose, logger):
        if _keras_cb is not object:
            super().__init__()
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.logger = logger
        self.last_printed = -1

    def on_epoch_end(self, epoch, logs=None):
        current = epoch + 1
        metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in logs.items()) if logs else ""

        if self.verbose < 1:
            bucket = int((current / self.n_epochs) / self.verbose)
            if bucket > self.last_printed:
                self.last_printed = bucket
                self.logger.adhoc_progress(current, self.n_epochs, metrics_str or None)
        else:
            interval = int(self.verbose)
            if current % interval == 0 or current == self.n_epochs:
                self.logger.adhoc_progress(current, self.n_epochs, metrics_str or None)


class NNAdapter(ModelAdapter):

    def get_gpu_usage(self, params):
        gpu = (params or {}).get('gpu', 'auto')
        if gpu != 'auto':
            raise ValueError(f"NNAdapter only supports gpu='auto', got {gpu!r}")
        return GPU_POSSIBLE

    def get_params(self, params, logger=None):
        if params is None:
            return {}
        params = params.copy()
        gpu = params.pop('gpu', 'auto')
        if gpu != 'auto':
            raise ValueError(f"NNAdapter only supports gpu='auto', got {gpu!r}")
        return params

    def get_fit_params(self, data_dict, params=None, logger=None):
        from .._data_wrapper import unwrap

        fit_params = super().get_fit_params(data_dict, params, logger)

        train_v_X = data_dict.get('X', (None, None))[1]
        train_v_y = data_dict.get('y', (None, None))[1]

        if self.eval_mode and self.eval_mode != 'none' and train_v_X is not None and train_v_y is not None:
            fit_params['eval_set'] = [(unwrap(train_v_X), unwrap(train_v_y))]

        if self.verbose > 0 and logger is not None and tf is not None:
            n_epochs = (params or {}).get('epochs', 100)
            fit_params['callbacks'] = [_ProgressCallback(n_epochs, self.verbose, logger)]

        return fit_params

    @staticmethod
    def _get_evals_result(processor):
        obj = processor.obj
        evals_result = getattr(obj, 'evals_result_', {})
        if not evals_result:
            return pd.DataFrame()
        return pd.concat(
            [pd.DataFrame(v).stack().rename(k) for k, v in evals_result.items()], axis=1
        ).stack()


NNAdapter.result_objs = {
    'evals_result': (NNAdapter._get_evals_result, True),
}

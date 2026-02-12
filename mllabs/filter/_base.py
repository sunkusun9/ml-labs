import numpy as np


class DataFilter:
    def __call__(self, data_dict):
        indices = self._select(data_dict)
        return {key: val.iloc(indices) for key, val in data_dict.items()}

    def _select(self, data_dict):
        raise NotImplementedError

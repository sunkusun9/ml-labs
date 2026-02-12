import numpy as np

from ._base import DataFilter


class IndexFilter(DataFilter):
    def __init__(self, index):
        self.index = index

    def _select(self, data_dict):
        first_val = next(iter(data_dict.values()))
        data_index = first_val.get_index()
        mask = np.isin(data_index, self.index)
        return np.where(mask)[0]

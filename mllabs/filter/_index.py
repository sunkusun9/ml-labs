import numpy as np

from ._base import DataFilter


class IndexFilter(DataFilter):
    """Select rows whose index values appear in a provided index array.

    Args:
        index (array-like): The set of index values to keep.
    """

    def __init__(self, index):
        self.index = index

    def _select(self, data_dict):
        first_val = next(iter(data_dict.values()))
        data_index = first_val.get_index()
        mask = np.isin(data_index, self.index)
        return np.where(mask)[0]

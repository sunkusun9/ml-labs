import numpy as np


class DataFilter:
    """Base class for data filters used with :class:`~mllabs.collector.SHAPCollector`.

    Subclasses implement :meth:`_select` to return a row index array.
    The filter is applied to all arrays in the input ``data_dict`` identically.
    """

    def __call__(self, data_dict):
        """Apply the filter to every array in *data_dict*.

        Args:
            data_dict (dict): ``{key: DataWrapper}`` mapping.

        Returns:
            dict: Filtered ``{key: DataWrapper}`` with rows selected by
            :meth:`_select`.
        """
        indices = self._select(data_dict)
        return {key: val.iloc(indices) for key, val in data_dict.items()}

    def _select(self, data_dict):
        raise NotImplementedError

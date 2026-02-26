import numpy as np

from ._base import DataFilter


class RandomFilter(DataFilter):
    """Randomly subsample rows from a data dict.

    Args:
        n (int, optional): Absolute number of rows to sample.
            Mutually exclusive with *frac*.
        frac (float, optional): Fraction of rows to sample (0–1).
            Mutually exclusive with *n*.
        random_state (int, optional): Random seed for reproducibility.
    """

    def __init__(self, n=None, frac=None, random_state=None):
        if n is not None and frac is not None:
            raise ValueError("n and frac are mutually exclusive")
        self.n = n
        self.frac = frac
        self.random_state = random_state

    def _select(self, data_dict):
        first_val = next(iter(data_dict.values()))
        total = first_val.get_shape()[0]

        if self.n is not None:
            sample_n = min(self.n, total)
        elif self.frac is not None:
            sample_n = int(total * self.frac)
        else:
            return np.arange(total)

        rng = np.random.RandomState(self.random_state)
        return np.sort(rng.choice(total, size=sample_n, replace=False))

from ._categorical import CategoricalConverter, CategoricalPairCombiner, FrequencyEncoder

__all__ = [
    "CategoricalConverter",
    "CategoricalPairCombiner",
    "FrequencyEncoder",
]

try:
    from ._polars import PolarsLoader, ExprProcessor
    from ._pandas import PandasConverter
    __all__.extend(["PolarsLoader", "ExprProcessor", "PandasConverter"])
except ImportError:
    pass

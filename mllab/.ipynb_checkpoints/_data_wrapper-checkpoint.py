"""
Data Wrapper for supporting multiple data libraries (pandas, Polars, cudf, numpy)

Provides a unified interface for common DataFrame operations across different libraries.
"""

from abc import ABC, abstractmethod
import numpy as np
from ._aggregator import Aggregator, PandasAggregator


class DataWrapper(ABC):
    """Abstract base class for data wrappers

    Provides unified interface for common operations across different data libraries.
    """

    def __init__(self, data):
        """Initialize wrapper with native data object"""
        self.data = data

    @abstractmethod
    def iloc(self, indices):
        """Select rows by integer indices

        Args:
            indices: Integer indices (list, array, or slice)

        Returns:
            DataWrapper: New wrapper with selected rows
        """
        pass

    @abstractmethod
    def select_columns(self, columns):
        """Select specific columns

        Args:
            columns: Column names (list or single column name)

        Returns:
            DataWrapper: New wrapper with selected columns
        """
        pass

    @abstractmethod
    def get_columns(self):
        """Get list of column names

        Returns:
            list: List of column names
        """
        pass

    @abstractmethod
    def get_shape(self):
        """Get shape of data

        Returns:
            tuple: (n_rows, n_cols)
        """
        pass

    @abstractmethod
    def get_index(self):
        """Get index of data

        Returns:
            Index object (type depends on library)
        """
        pass

    @staticmethod
    @abstractmethod
    def concat(wrappers, axis=1):
        """Concatenate multiple wrappers

        Args:
            wrappers: List of DataWrapper objects
            axis: Axis to concatenate (0=rows, 1=columns)

        Returns:
            DataWrapper: Concatenated result
        """
        pass

    def to_native(self):
        """Get native data object

        Returns:
            Native data object (DataFrame, ndarray, etc.)
        """
        return self.data

    @staticmethod
    def from_native(data):
        """Create appropriate wrapper from native data object

        Args:
            data: Native data object

        Returns:
            DataWrapper: Appropriate wrapper instance
        """
        if data is None:
            return None

        # Check type and return appropriate wrapper
        type_name = type(data).__name__
        module_name = type(data).__module__

        if 'pandas' in module_name:
            return PandasWrapper(data)
        elif 'polars' in module_name:
            return PolarsWrapper(data)
        elif 'cudf' in module_name:
            return CudfWrapper(data)
        elif isinstance(data, np.ndarray):
            return NumpyWrapper(data)
        else:
            # Default to pandas if unknown
            import pandas as pd
            if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
                return PandasWrapper(data)
            raise TypeError(f"Unsupported data type: {type(data)}")

    @classmethod
    @abstractmethod
    def from_output(cls, output, column_names=None, index=None):
        """Create wrapper instance from processor output with optional column names and index

        Processor의 출력(numpy array, DataFrame 등)을 받아서 해당 wrapper의
        native 타입으로 변환하고 컬럼명과 인덱스를 설정한 후 wrapper 인스턴스를 반환합니다.

        Args:
            output: Processor의 출력 (numpy array, DataFrame, etc.)
            column_names: 설정할 컬럼명 리스트 (None이면 자동 생성 또는 유지)
            index: 설정할 인덱스 (None이면 기본 인덱스 또는 유지)

        Returns:
            DataWrapper: Wrapper 인스턴스
        """
        pass

    def __getitem__(self, key):
        """Support indexing syntax: wrapper[key]"""
        if isinstance(key, (list, str)):
            return self.select_columns(key)
        else:
            # For other types, delegate to native object
            return DataWrapper.from_native(self.data[key])

    @staticmethod
    def get_aggregator():
        return Aggregator


class PandasWrapper(DataWrapper):
    """Wrapper for pandas DataFrame/Series"""

    def iloc(self, indices):
        return PandasWrapper(self.data.iloc[indices])

    def select_columns(self, columns):
        return PandasWrapper(self.data[columns])

    def get_columns(self):
        if hasattr(self.data, 'columns'):
            return self.data.columns.tolist()
        else:
            # Series has name
            return [self.data.name] if self.data.name is not None else [0]

    def get_shape(self):
        return self.data.shape

    def get_index(self):
        return self.data.index

    @staticmethod
    def concat(wrappers, axis=1):
        import pandas as pd
        native_list = [w.to_native() for w in wrappers]
        result = pd.concat(native_list, axis=axis)
        return PandasWrapper(result)

    @classmethod
    def from_output(cls, output, column_names=None, index=None):
        """Convert processor output to PandasWrapper

        Args:
            output: Processor 출력 (numpy array, DataFrame, etc.)
            column_names: 컬럼명 리스트
            index: 인덱스

        Returns:
            PandasWrapper 인스턴스
        """
        import pandas as pd

        if output is None:
            return None

        # 이미 pandas DataFrame인 경우
        if isinstance(output, pd.DataFrame):
            if column_names is not None:
                output.columns = column_names
            if index is not None:
                output.index = index
            return cls(output)

        # numpy array인 경우 → DataFrame으로 변환
        elif isinstance(output, np.ndarray):
            if output.ndim == 1:
                if column_names is None:
                    column_names = [0]
                df = pd.DataFrame({column_names[0]: output}, index=index)
            else:
                if column_names is None:
                    column_names = list(range(output.shape[1]))
                df = pd.DataFrame(output, columns=column_names, index=index)
            return cls(df)

        # 기타 DataFrame (polars, cudf) → pandas로 변환
        elif hasattr(output, 'to_pandas'):
            df = output.to_pandas()
            if column_names is not None:
                df.columns = column_names
            if index is not None:
                df.index = index
            return cls(df)

        # 변환 불가능한 경우 에러 발생
        else:
            raise TypeError(f"Cannot convert {type(output)} to pandas DataFrame")
    @staticmethod
    def get_aggregator():
        return Aggregator.PandasAggregator

class PolarsWrapper(DataWrapper):
    """Wrapper for Polars DataFrame"""

    def iloc(self, indices):
        import polars as pl
        if isinstance(indices, slice):
            result = self.data[indices]
        elif isinstance(indices, (list, np.ndarray)):
            result = self.data[indices]
        else:
            result = self.data[indices]
        return PolarsWrapper(result)

    def select_columns(self, columns):
        return PolarsWrapper(self.data[columns])

    def get_columns(self):
        return self.data.columns

    def get_shape(self):
        return self.data.shape

    def get_index(self):
        # Polars doesn't have index, return range
        return range(len(self.data))

    @staticmethod
    def concat(wrappers, axis=1):
        import polars as pl
        native_list = [w.to_native() for w in wrappers]
        if axis == 1:
            result = pl.concat(native_list, how="horizontal")
        else:
            result = pl.concat(native_list, how="vertical")
        return PolarsWrapper(result)

    @classmethod
    def from_output(cls, output, column_names=None, index=None):
        """Convert processor output to PolarsWrapper

        Args:
            output: Processor 출력 (numpy array, DataFrame, etc.)
            column_names: 컬럼명 리스트
            index: 인덱스 (polars는 인덱스 개념이 없으므로 무시됨)

        Returns:
            PolarsWrapper 인스턴스
        """
        import polars as pl

        if output is None:
            return None

        # 이미 polars DataFrame인 경우
        if isinstance(output, pl.DataFrame):
            if column_names is not None:
                output.columns = column_names
            return cls(output)

        # numpy array인 경우 → polars DataFrame으로 변환
        elif isinstance(output, np.ndarray):
            if output.ndim == 1:
                if column_names is None:
                    column_names = [0]
                df = pl.DataFrame({str(column_names[0]): output})
            else:
                if column_names is None:
                    column_names = [str(i) for i in range(output.shape[1])]
                else:
                    column_names = [str(c) for c in column_names]
                df = pl.DataFrame(output, schema=column_names)
            return cls(df)

        # pandas DataFrame → polars로 변환
        elif hasattr(output, 'to_numpy') and hasattr(output, 'columns'):
            df = pl.from_pandas(output)
            if column_names is not None:
                df.columns = column_names
            return cls(df)

        # 변환 불가능한 경우 에러 발생
        else:
            raise TypeError(f"Cannot convert {type(output)} to polars DataFrame")


class CudfWrapper(DataWrapper):
    """Wrapper for cuDF DataFrame (GPU-accelerated)"""

    def iloc(self, indices):
        return CudfWrapper(self.data.iloc[indices])

    def select_columns(self, columns):
        return CudfWrapper(self.data[columns])

    def get_columns(self):
        if hasattr(self.data, 'columns'):
            return self.data.columns.tolist()
        else:
            return [self.data.name] if self.data.name is not None else [0]

    def get_shape(self):
        return self.data.shape

    def get_index(self):
        return self.data.index

    @staticmethod
    def concat(wrappers, axis=1):
        import cudf
        native_list = [w.to_native() for w in wrappers]
        result = cudf.concat(native_list, axis=axis)
        return CudfWrapper(result)

    @classmethod
    def from_output(cls, output, column_names=None, index=None):
        """Convert processor output to CudfWrapper

        Args:
            output: Processor 출력 (numpy array, DataFrame, etc.)
            column_names: 컬럼명 리스트
            index: 인덱스

        Returns:
            CudfWrapper 인스턴스
        """
        import cudf

        if output is None:
            return None

        # 이미 cudf DataFrame인 경우
        if isinstance(output, cudf.DataFrame):
            if column_names is not None:
                output.columns = column_names
            if index is not None:
                output.index = index
            return cls(output)

        # numpy array인 경우 → cudf DataFrame으로 변환
        elif isinstance(output, np.ndarray):
            if output.ndim == 1:
                if column_names is None:
                    column_names = [0]
                df = cudf.DataFrame({column_names[0]: output}, index=index)
            else:
                if column_names is None:
                    column_names = list(range(output.shape[1]))
                df = cudf.DataFrame(output, columns=column_names, index=index)
            return cls(df)

        # pandas DataFrame → cudf로 변환
        elif hasattr(output, 'to_numpy') and hasattr(output, 'columns'):
            df = cudf.from_pandas(output)
            if column_names is not None:
                df.columns = column_names
            if index is not None:
                df.index = index
            return cls(df)

        # 변환 불가능한 경우 에러 발생
        else:
            raise TypeError(f"Cannot convert {type(output)} to cudf DataFrame")


class NumpyWrapper(DataWrapper):
    """Wrapper for NumPy ndarray"""

    def __init__(self, data, columns=None):
        """Initialize with ndarray and optional column names

        Args:
            data: numpy ndarray
            columns: Optional list of column names
        """
        super().__init__(data)
        if columns is None:
            # Generate default column names
            if data.ndim == 1:
                self.columns = [0]
            else:
                self.columns = list(range(data.shape[1]))
        else:
            self.columns = columns

    def iloc(self, indices):
        if isinstance(indices, slice):
            result = self.data[indices]
        elif isinstance(indices, (list, np.ndarray)):
            result = self.data[indices]
        else:
            result = self.data[indices]
        return NumpyWrapper(result, self.columns)

    def select_columns(self, columns):
        if isinstance(columns, str):
            # Single column name
            if columns in self.columns:
                col_idx = self.columns.index(columns)
                if self.data.ndim == 1:
                    return NumpyWrapper(self.data, [columns])
                else:
                    return NumpyWrapper(self.data[:, col_idx], [columns])
            else:
                raise KeyError(f"Column '{columns}' not found")
        else:
            # Multiple columns
            col_indices = [self.columns.index(c) for c in columns]
            if self.data.ndim == 1:
                return NumpyWrapper(self.data, columns)
            else:
                return NumpyWrapper(self.data[:, col_indices], columns)

    def get_columns(self):
        return self.columns

    def get_shape(self):
        return self.data.shape

    def get_index(self):
        # NumPy doesn't have index, return range
        return range(len(self.data))

    @staticmethod
    def concat(wrappers, axis=1):
        native_list = [w.to_native() for w in wrappers]
        result = np.concatenate(native_list, axis=axis)

        # Merge column names
        if axis == 1:
            all_columns = []
            for w in wrappers:
                all_columns.extend(w.columns)
            return NumpyWrapper(result, all_columns)
        else:
            # Use first wrapper's columns for vertical concat
            return NumpyWrapper(result, wrappers[0].columns)

    @classmethod
    def from_output(cls, output, column_names=None, index=None):
        """Convert processor output to NumpyWrapper

        Args:
            output: Processor 출력 (numpy array, DataFrame, etc.)
            column_names: 컬럼명 리스트
            index: 인덱스 (numpy는 인덱스 개념이 없으므로 무시됨)

        Returns:
            NumpyWrapper 인스턴스
        """
        if output is None:
            return None

        # 이미 numpy array인 경우
        if isinstance(output, np.ndarray):
            if column_names is None:
                if output.ndim == 1:
                    column_names = [0]
                else:
                    column_names = list(range(output.shape[1]))
            return cls(output, column_names)

        # DataFrame인 경우 → numpy array로 변환
        elif hasattr(output, 'to_numpy'):
            arr = output.to_numpy()
            if column_names is None:
                if hasattr(output, 'columns'):
                    column_names = output.columns.tolist()
                else:
                    column_names = list(range(arr.shape[1] if arr.ndim > 1 else 1))
            return cls(arr, column_names)

        # 변환 불가능한 경우 에러 발생
        else:
            raise TypeError(f"Cannot convert {type(output)} to numpy array")


def wrap(data):
    """Convenience function to wrap native data

    Args:
        data: Native data object

    Returns:
        DataWrapper: Appropriate wrapper instance
    """
    return DataWrapper.from_native(data)


def unwrap(wrapper):
    """Convenience function to unwrap to native data

    Args:
        wrapper: DataWrapper instance

    Returns:
        Native data object
    """
    if wrapper is None:
        return None
    if isinstance(wrapper, DataWrapper):
        return wrapper.to_native()
    else:
        # Already native
        return wrapper

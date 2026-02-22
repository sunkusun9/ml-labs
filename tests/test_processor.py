import pytest
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

requires_polars = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

from mllabs.processor import (
    CategoricalConverter,
    CategoricalPairCombiner,
    FrequencyEncoder,
)

if HAS_POLARS:
    from mllabs.processor import (
        PolarsLoader,
        ExprProcessor,
        PandasConverter,
    )


@pytest.fixture
def sample_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b,c\n1,2.5,x\n2,3.5,y\n3,4.5,z\n")
    return str(csv_path)


@pytest.fixture
def sample_csv_list(tmp_path):
    csv1 = tmp_path / "sample1.csv"
    csv2 = tmp_path / "sample2.csv"
    csv1.write_text("a,b,c\n1,2.5,x\n2,3.5,y\n")
    csv2.write_text("a,b,c\n3,4.5,z\n4,5.5,w\n")
    return [str(csv1), str(csv2)]


@pytest.fixture
def sample_polars_df():
    if not HAS_POLARS:
        pytest.skip("polars not installed")
    return pl.DataFrame({
        "a": [1, 2, 3],
        "b": [2.5, 3.5, 4.5],
        "c": ["x", "y", "z"],
    })


@pytest.fixture
def sample_pandas_df():
    return pd.DataFrame({
        "a": [1, 2, 3],
        "b": [2.5, 3.5, 4.5],
        "c": ["x", "y", "z"],
    })


@requires_polars
class TestPolarsLoader:
    def test_fit_single_csv(self, sample_csv):
        loader = PolarsLoader()
        loader.fit(sample_csv)
        assert hasattr(loader, 'df_type_')
        assert hasattr(loader, 'pl_type_')
        assert loader.fitted_ is True

    def test_fit_csv_list(self, sample_csv_list):
        loader = PolarsLoader()
        loader.fit(sample_csv_list)
        assert loader.fitted_ is True

    def test_transform_single_csv(self, sample_csv):
        loader = PolarsLoader()
        loader.fit(sample_csv)
        result = loader.transform(sample_csv)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert result.columns == ["a", "b", "c"]

    def test_transform_csv_list(self, sample_csv_list):
        loader = PolarsLoader()
        loader.fit(sample_csv_list)
        result = loader.transform(sample_csv_list)
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 4

    def test_get_feature_names_out(self, sample_csv):
        loader = PolarsLoader()
        loader.fit(sample_csv)
        names = loader.get_feature_names_out()
        assert "a" in names
        assert "b" in names
        assert "c" in names

    def test_get_params(self):
        loader = PolarsLoader(predefined_types={"a": pl.Int32}, read_method="read_csv")
        params = loader.get_params()
        assert params["predefined_types"] == {"a": pl.Int32}
        assert params["read_method"] == "read_csv"

    def test_predefined_types(self, sample_csv):
        loader = PolarsLoader(predefined_types={"a": pl.Int32})
        loader.fit(sample_csv)
        result = loader.transform(sample_csv)
        assert result["a"].dtype == pl.Int32


@requires_polars
class TestExprProcessor:
    def test_with_columns_true(self, sample_polars_df):
        proc = ExprProcessor(dict_expr={"d": pl.col("a") * 2}, with_columns=True)
        proc.fit(sample_polars_df)
        result = proc.transform(sample_polars_df)
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns
        assert "d" in result.columns
        assert result["d"].to_list() == [2, 4, 6]

    def test_with_columns_false(self, sample_polars_df):
        proc = ExprProcessor(dict_expr={"d": pl.col("a") * 2}, with_columns=False)
        proc.fit(sample_polars_df)
        result = proc.transform(sample_polars_df)
        assert "d" in result.columns
        assert len(result.columns) == 1

    def test_get_feature_names_out(self, sample_polars_df):
        proc = ExprProcessor(dict_expr={"d": pl.col("a") * 2}, with_columns=True)
        proc.fit(sample_polars_df)
        names = proc.get_feature_names_out()
        assert "a" in names
        assert "b" in names
        assert "c" in names
        assert "d" in names

    def test_get_params(self):
        expr = {"d": pl.col("a") * 2}
        proc = ExprProcessor(dict_expr=expr, with_columns=True)
        params = proc.get_params()
        assert params["dict_expr"] == expr
        assert params["with_columns"] is True


@requires_polars
class TestPandasConverter:
    def test_basic_conversion(self, sample_polars_df):
        conv = PandasConverter()
        conv.fit(sample_polars_df)
        result = conv.transform(sample_polars_df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b", "c"]

    def test_with_index_col(self, sample_polars_df):
        conv = PandasConverter(index_col="a")
        conv.fit(sample_polars_df)
        result = conv.transform(sample_polars_df)
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "a"
        assert "a" not in result.columns

    def test_get_feature_names_out(self, sample_polars_df):
        conv = PandasConverter(index_col="a")
        conv.fit(sample_polars_df)
        names = conv.get_feature_names_out()
        assert "a" not in names
        assert "b" in names
        assert "c" in names

    def test_get_params(self):
        conv = PandasConverter(index_col="id")
        params = conv.get_params()
        assert params["index_col"] == "id"


class TestCategoricalConverter:
    def test_pandas_all_columns(self, sample_pandas_df):
        conv = CategoricalConverter()
        conv.fit(sample_pandas_df)
        result = conv.transform(sample_pandas_df)
        for col in result.columns:
            assert result[col].dtype.name == "category"

    def test_pandas_specific_columns(self, sample_pandas_df):
        conv = CategoricalConverter(columns=["c"])
        conv.fit(sample_pandas_df)
        result = conv.transform(sample_pandas_df)
        assert result["c"].dtype.name == "category"
        assert result["a"].dtype != "category"

    @requires_polars
    def test_polars_all_columns(self):
        df = pl.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
        conv = CategoricalConverter()
        conv.fit(df)
        result = conv.transform(df)
        for col in result.columns:
            assert result[col].dtype == pl.Categorical

    @requires_polars
    def test_polars_specific_columns(self, sample_polars_df):
        conv = CategoricalConverter(columns=["c"])
        conv.fit(sample_polars_df)
        result = conv.transform(sample_polars_df)
        assert result["c"].dtype == pl.Categorical
        assert result["a"].dtype != pl.Categorical

    def test_numpy(self):
        arr = np.array([[1, "a"], [2, "b"], [3, "c"]], dtype=object)
        conv = CategoricalConverter(columns=[1])
        conv.fit(arr)
        result = conv.transform(arr)
        assert result.dtype == object

    def test_get_feature_names_out(self, sample_pandas_df):
        conv = CategoricalConverter(columns=["a", "c"])
        conv.fit(sample_pandas_df)
        names = conv.get_feature_names_out()
        assert "a" in names
        assert "c" in names

    def test_column_index(self, sample_pandas_df):
        conv = CategoricalConverter(columns=[2])
        conv.fit(sample_pandas_df)
        result = conv.transform(sample_pandas_df)
        assert result["c"].dtype.name == "category"


class TestCategoricalPairCombiner:
    @pytest.fixture
    def pair_pandas_df(self):
        return pd.DataFrame({
            "cat1": ["a", "a", "b", "b", "c"],
            "cat2": ["x", "x", "y", "y", "z"],
            "val": [1, 2, 3, 4, 5],
        })

    @pytest.fixture
    def pair_polars_df(self):
        if not HAS_POLARS:
            pytest.skip("polars not installed")
        return pl.DataFrame({
            "cat1": ["a", "a", "b", "b", "c"],
            "cat2": ["x", "x", "y", "y", "z"],
            "val": [1, 2, 3, 4, 5],
        })

    def test_pandas_basic(self, pair_pandas_df):
        comb = CategoricalPairCombiner(pairs=[("cat1", "cat2")], min_frequency=0)
        comb.fit(pair_pandas_df)
        result = comb.transform(pair_pandas_df)
        assert "cat1__cat2" in result.columns
        assert result["cat1__cat2"].tolist() == ["a__x", "a__x", "b__y", "b__y", "c__z"]

    @requires_polars
    def test_polars_basic(self, pair_polars_df):
        comb = CategoricalPairCombiner(pairs=[("cat1", "cat2")], min_frequency=0)
        comb.fit(pair_polars_df)
        result = comb.transform(pair_polars_df)
        assert "cat1__cat2" in result.columns
        assert result["cat1__cat2"].to_list() == ["a__x", "a__x", "b__y", "b__y", "c__z"]

    def test_min_frequency_filter(self, pair_pandas_df):
        comb = CategoricalPairCombiner(pairs=[("cat1", "cat2")], min_frequency=1)
        comb.fit(pair_pandas_df)
        result = comb.transform(pair_pandas_df)
        values = result["cat1__cat2"].tolist()
        assert values[:4] == ["a__x", "a__x", "b__y", "b__y"]
        assert pd.isna(values[4])

    def test_drop_original(self, pair_pandas_df):
        comb = CategoricalPairCombiner(pairs=[("cat1", "cat2")], min_frequency=0, drop_original=True)
        comb.fit(pair_pandas_df)
        result = comb.transform(pair_pandas_df)
        assert "cat1" not in result.columns
        assert "cat2" not in result.columns
        assert "cat1__cat2" in result.columns
        assert "val" in result.columns

    def test_custom_sep(self, pair_pandas_df):
        comb = CategoricalPairCombiner(pairs=[("cat1", "cat2")], min_frequency=0, sep="-")
        comb.fit(pair_pandas_df)
        result = comb.transform(pair_pandas_df)
        assert "cat1-cat2" in result.columns
        assert result["cat1-cat2"].tolist()[0] == "a-x"

    def test_custom_new_col_names(self, pair_pandas_df):
        comb = CategoricalPairCombiner(pairs=[("cat1", "cat2")], min_frequency=0, new_col_names=["combined"])
        comb.fit(pair_pandas_df)
        result = comb.transform(pair_pandas_df)
        assert "combined" in result.columns

    def test_missing_value_handling(self):
        df = pd.DataFrame({
            "cat1": ["a", None, "b"],
            "cat2": ["x", "y", None],
        })
        comb = CategoricalPairCombiner(pairs=[("cat1", "cat2")], min_frequency=0, missing_value="MISSING")
        comb.fit(df)
        result = comb.transform(df)
        assert result["cat1__cat2"].tolist() == ["a__x", "MISSING", "MISSING"]

    def test_numpy_basic(self):
        arr = np.array([["a", "x"], ["a", "x"], ["b", "y"]], dtype=object)
        comb = CategoricalPairCombiner(pairs=[(0, 1)], min_frequency=0)
        comb.fit(arr)
        result = comb.transform(arr)
        assert result.shape[1] == 3
        assert result[0, 2] == "a__x"

    def test_multiple_pairs(self, pair_pandas_df):
        df = pair_pandas_df.copy()
        df["cat3"] = ["p", "q", "p", "q", "p"]
        comb = CategoricalPairCombiner(pairs=[("cat1", "cat2"), ("cat1", "cat3")], min_frequency=0)
        comb.fit(df)
        result = comb.transform(df)
        assert "cat1__cat2" in result.columns
        assert "cat1__cat3" in result.columns

    def test_not_fitted_error(self, pair_pandas_df):
        comb = CategoricalPairCombiner(pairs=[("cat1", "cat2")], min_frequency=0)
        with pytest.raises(RuntimeError):
            comb.transform(pair_pandas_df)


class TestFrequencyEncoder:
    @pytest.fixture
    def freq_pandas_df(self):
        return pd.DataFrame({'ST depression': [0.0, 0.0, 0.0, 1.2, 1.2, 3.5]})

    @pytest.fixture
    def freq_polars_df(self):
        if not HAS_POLARS:
            pytest.skip("polars not installed")
        return pl.DataFrame({'ST depression': [0.0, 0.0, 0.0, 1.2, 1.2, 3.5]})

    def test_pandas_output_columns(self, freq_pandas_df):
        fe = FrequencyEncoder()
        fe.fit(freq_pandas_df)
        result = fe.transform(freq_pandas_df)
        assert list(result.columns) == ['ST depression_freq']

    def test_pandas_normalize(self, freq_pandas_df):
        fe = FrequencyEncoder(normalize=True)
        fe.fit(freq_pandas_df)
        result = fe.transform(freq_pandas_df)
        assert abs(result['ST depression_freq'].iloc[0] - 3/6) < 1e-6
        assert abs(result['ST depression_freq'].iloc[3] - 2/6) < 1e-6
        assert abs(result['ST depression_freq'].iloc[5] - 1/6) < 1e-6

    def test_pandas_count(self, freq_pandas_df):
        fe = FrequencyEncoder(normalize=False)
        fe.fit(freq_pandas_df)
        result = fe.transform(freq_pandas_df)
        assert result['ST depression_freq'].iloc[0] == 3
        assert result['ST depression_freq'].iloc[3] == 2
        assert result['ST depression_freq'].iloc[5] == 1

    def test_pandas_unseen_value(self, freq_pandas_df):
        fe = FrequencyEncoder()
        fe.fit(freq_pandas_df)
        unseen = pd.DataFrame({'ST depression': [99.9]})
        result = fe.transform(unseen)
        assert result['ST depression_freq'].iloc[0] == 0.0

    def test_pandas_multiple_columns(self):
        df = pd.DataFrame({'a': [1, 1, 2], 'b': [3, 3, 3]})
        fe = FrequencyEncoder()
        fe.fit(df)
        result = fe.transform(df)
        assert list(result.columns) == ['a_freq', 'b_freq']

    def test_get_feature_names_out(self, freq_pandas_df):
        fe = FrequencyEncoder()
        fe.fit(freq_pandas_df)
        names = fe.get_feature_names_out()
        assert names == ['ST depression_freq']

    def test_get_feature_names_out_with_input(self, freq_pandas_df):
        fe = FrequencyEncoder()
        fe.fit(freq_pandas_df)
        names = fe.get_feature_names_out(['ST depression'])
        assert names == ['ST depression_freq']

    @requires_polars
    def test_polars_output_columns(self, freq_polars_df):
        fe = FrequencyEncoder()
        fe.fit(freq_polars_df)
        result = fe.transform(freq_polars_df)
        assert result.columns == ['ST depression_freq']

    @requires_polars
    def test_polars_normalize(self, freq_polars_df):
        fe = FrequencyEncoder(normalize=True)
        fe.fit(freq_polars_df)
        result = fe.transform(freq_polars_df)
        assert abs(result['ST depression_freq'][0] - 3/6) < 1e-6
        assert abs(result['ST depression_freq'][3] - 2/6) < 1e-6

    @requires_polars
    def test_polars_unseen_value(self, freq_polars_df):
        fe = FrequencyEncoder()
        fe.fit(freq_polars_df)
        unseen = pl.DataFrame({'ST depression': [99.9]})
        result = fe.transform(unseen)
        assert result['ST depression_freq'][0] == 0.0

    def test_numpy_output_shape(self):
        arr = np.array([[0.0], [0.0], [0.0], [1.2], [1.2], [3.5]])
        fe = FrequencyEncoder()
        fe.fit(arr)
        result = fe.transform(arr)
        assert result.shape == (6, 1)

    def test_numpy_normalize(self):
        arr = np.array([[0.0], [0.0], [0.0], [1.2], [1.2], [3.5]])
        fe = FrequencyEncoder(normalize=True)
        fe.fit(arr)
        result = fe.transform(arr)
        assert abs(result[0, 0] - 3/6) < 1e-6
        assert abs(result[3, 0] - 2/6) < 1e-6

    def test_numpy_unseen_value(self):
        arr = np.array([[0.0], [1.0], [2.0]])
        fe = FrequencyEncoder()
        fe.fit(arr)
        unseen = np.array([[99.0]])
        result = fe.transform(unseen)
        assert result[0, 0] == 0.0

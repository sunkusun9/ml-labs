# Processors

ml-labs provides a set of built-in sklearn-compatible processors for common preprocessing tasks.

## Built-in Processors

### CatConverter

Converts specified columns to a categorical dtype (pandas `category` / polars `Categorical`).

```python
from mllabs.processor import CatConverter

CatConverter(columns=None)   # None → convert all columns
CatConverter(columns=['city', 'gender'])
```

Output columns are the same as input columns with dtype changed. Supports pandas, polars, and numpy inputs.

### CatOOVFilter

Filters out-of-vocabulary (OOV) values in categorical columns. Values not seen during `fit()` (or seen fewer than `min_frequency` times) are replaced with `missing_value`.

```python
from mllabs.processor import CatOOVFilter

CatOOVFilter(
    columns=None,                    # None → all columns
    min_frequency=0,                 # values with count <= this are treated as OOV
    missing_value=None,              # replacement for OOV/missing values
    treat_empty_string_as_missing=True,
)
```

Fitted attribute: `categories_` — dict of `{column: [allowed_values]}`.

### CatPairCombiner

Combines pairs of categorical columns into new interaction columns by concatenating their string representations.

```python
from mllabs.processor import CatPairCombiner

CatPairCombiner(
    pairs=[('city', 'gender'), ('age_bin', 'city')],
    sep='__',
    treat_empty_string_as_missing=True,
    new_col_names=None,   # None → 'city__gender', 'age_bin__city'
)
```

Output column names default to `colA{sep}colB`. Either missing value in a pair produces `None` in the output. Supports integer column indices for numpy inputs.

### FrequencyEncoder

Replaces each categorical value with its frequency (proportion or count) observed during `fit()`.

```python
from mllabs.processor import FrequencyEncoder

FrequencyEncoder(normalize=True)   # True → proportion, False → count
```

Output column names are `{col}_freq`. Unseen values at transform time receive `0`.

---

## Polars-Optional Processors

Available only when `polars` is installed (`pip install ml-labs[polars]`).

### PolarsLoader

Reads one or more CSV files into a Polars DataFrame with automatically optimized dtypes. Uses `get_type_df` internally to analyze data ranges.

```python
from mllabs.processor import PolarsLoader

PolarsLoader(
    predefined_types={'id': pl.Int64},   # override specific column types
    read_method='read_csv',              # any pl.read_* method name
)

# fit: accepts a file path (str) or list of file paths
# transform: returns pl.DataFrame with optimized schema
```

### ExprProcessor

Applies a dict of Polars expressions to a DataFrame via `with_columns` or `select`.

```python
from mllabs.processor import ExprProcessor
import polars as pl

ExprProcessor(
    dict_expr={
        'log_income': pl.col('income').log1p(),
        'age_sq': pl.col('age') ** 2,
    },
    with_columns=True,   # True → add/replace columns; False → select only these columns
)
```

When `with_columns=True`, output includes all original columns plus the new/replaced ones. When `False`, output contains only the expressions defined in `dict_expr`.

### PandasConverter

Converts a Polars DataFrame to pandas. Useful as the final stage in a Polars-based pipeline.

```python
from mllabs.processor import PandasConverter

PandasConverter(index_col=None)   # optionally set a column as the DataFrame index
```

---

## Type Utilities

Utilities for analyzing column types and generating optimal dtype mappings. Useful for preprocessing pipelines that need to minimize memory usage.

```python
from mllabs.processor import get_type_df, get_type_pl, get_type_pd, merge_type_df
```

### get_type_df(df)

Analyzes a Polars DataFrame (or LazyFrame) and returns a `pd.DataFrame` with per-column statistics and dtype-fit flags.

```python
df_type = pl.scan_csv('train.csv').pipe(get_type_df)
# Columns: min, max, na, count, n_unique, dtype, f32, i32, i16, i8
# f32/i32/i16/i8: True if numeric values fit within that type's range
```

### get_type_pl(df_type, ...) / get_type_pd(df_type, ...)

Convert the `df_type` analysis into a type mapping dict for Polars or pandas loading.

```python
pl_types = get_type_pl(
    df_type,
    predefine={'id': pl.Int64},  # always override these
    f32=True,                    # use Float32 where possible
    i64=False,                   # use smallest int type that fits
    cat_max=50,                  # columns with ≤50 unique values → Categorical
    txt_cols=['description'],    # force these columns to String
)

pd_types = get_type_pd(df_type, predefine={'id': 'int64'}, cat_max=50)
```

### merge_type_df(dfs)

Merges `df_type` results from multiple files (e.g., sharded datasets) into a single summary. Takes the global min/max across files to determine safe dtype ranges.

```python
from glob import glob

df_type = merge_type_df([
    pl.scan_csv(f).pipe(get_type_df) for f in glob('data/*.csv')
])
```

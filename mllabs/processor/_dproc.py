import os
import pandas as pd
import numpy as np
import polars as pl
import polars.selectors as cs
from functools import partial
from itertools import product
from collections import OrderedDict

def get_type_df(df):
    """
    get data type of polars df
    Parameters:
        df: pl.DataFrame, pl.LazyFrame
            DataFrame to examine the proper data type
    Returns:
        pd.DataFrame
            DataFrame which contains the information about the types
    Examples:
        >>> df_type = pl.scan_csv('test.csv').pipe(get_type_df)
    """
    stat = [
        ('min', cs.numeric().min()),
        ('max', cs.numeric().max()),
        ('na', cs.all().is_null().sum()),
        ('count', cs.all().count()),
        ('n_unique', cs.all().n_unique()),
    ]
    df_type = df.select(
        **{k: pl.struct(v) for k, v in stat}
    ).unpivot().unnest('value').rename({'variable': 'stat'})\
    .unpivot(index='stat', variable_name='feature')
    if type(df_type) == pl.LazyFrame:
        df_type = df_type.collect()
    if type(df) == pl.lazyframe.frame.LazyFrame:
        dtypes = df.collect_schema().dtypes()
        columns = df.collect_schema().names()
    else:
        dtypes = df.dtypes
        columns = df.columns
    df_type = df_type.pivot(
        index='feature', on='stat', values='value'
    ).to_pandas().assign(
        feature = lambda x: x['feature'].astype(str)
    ).set_index('feature').join(
        pd.Series([str(i) for i in dtypes], index=columns, name='dtype')
    )
    for i, mn, mx in [
        ('f32', np.finfo(np.float32).min, np.finfo(np.float32).max),
        ('i32', np.iinfo(np.int32).min, np.iinfo(np.int32).max),
        ('i16', np.iinfo(np.int16).min, np.iinfo(np.int16).max),
        ('i8', np.iinfo(np.int8).min, np.iinfo(np.int8).max)
    ]:
        numeric_mask = df_type['dtype'].str.startswith(('Int', 'Float'))
        df_type[i] = df_type.loc[numeric_mask].apply(
            lambda x: (x['min'] >= mn) and (x['max'] <= mx), axis=1
        ).reindex(df_type.index, fill_value=False)
    return df_type

def get_type_vars(var_list):
    """
    Processes a list of variables and generates a DataFrame summarizing their properties, such as 
    the number of unique values, count, and range (min, max). Also checks whether numeric variables 
    fit within certain data type ranges (e.g., float32, int32, int16, int8).

    Args:
        var_list (list of tuples): A list of tuples where each tuple contains the following elements:
            - src (str): The source or origin of the variable.
            - s (pd.Series): The pandas Series representing the variable.
            - desc (str): A description of the variable (unused in the function).

    Returns:
        pd.DataFrame: A pandas DataFrame indexed by variable name with the following columns:
            - 'src' (str): The source of the variable.
            - 'dtype' (str): The data type of the variable.
            - 'na' (int): The number of missing values in the variable.
            - 'n_unique' (int): The number of unique values in the variable.
            - 'count' (int): The total count of non-missing values.
            - 'min' (varied): The minimum value of the variable (if applicable).
            - 'max' (varied): The maximum value of the variable (if applicable).
            - 'f32' (bool): Whether the variable can fit within a float32 data type range.
            - 'i32' (bool): Whether the variable can fit within an int32 data type range.
            - 'i16' (bool): Whether the variable can fit within an int16 data type range.
            - 'i8' (bool): Whether the variable can fit within an int8 data type range.

    Example:
        >>> var_list = [
        ...     ('source1', pd.Series([1, 2, 3], dtype='int32'), 'description1', 'Numeric'),
        ...     ('source2', pd.Series(['a', 'b', 'a'], dtype='category'), 'description2', 'Categorical')
        ... ]
        >>> get_type_vars(var_list)
        # Returns a DataFrame with statistics on the variables, including dtype compatibility.
    """
    s_list = list()
    for src, s, desc in var_list:
        if str(s.dtype) == 'category':
            t = 'Categorical'
            if s.dtype.ordered:
                s_list.append(
                    pd.Series(
                        [src, s.name, desc, t, s.isna().sum()] + s.agg(['nunique', 'count', 'min', 'max']).tolist(),
                        index=['src', 'var', 'Description', 'dtype', 'na', 'n_unique', 'count', 'min', 'max']
                    )
                )
            else:
                s_list.append(
                    pd.Series(
                        [src, s.name, desc, t, s.isna().sum()] + s.agg(['nunique', 'count']).tolist(),
                        index=['src', 'var', 'Description', 'dtype', 'na', 'n_unique', 'count']
                    )
                )
        else:
            t = str(s.dtype)
            if t == 'str':
                t = 'String'
            if t == 'datetime64[ns]':
                t = 'Datetime'
            else:
                t = t[:1].upper() + t[1:]
            s_list.append(
                pd.Series(
                    [src, s.name, desc, s.isna().sum(), t] + s.agg(['nunique', 'count', 'min', 'max']).tolist(),
                    index=['src', 'var', 'Description', 'na', 'dtype', 'n_unique', 'count', 'min', 'max']
                )
            )
    df = pd.DataFrame(s_list).set_index('var')
    if 'min' in df.columns:
        for i, mn, mx in [
            ('f32', np.finfo(np.float32).min, np.finfo(np.float32).max),
            ('i32', np.iinfo(np.int32).min, np.iinfo(np.int32).max),
            ('i16', np.iinfo(np.int16).min, np.iinfo(np.int16).max),
            ('i8', np.iinfo(np.int8).min, np.iinfo(np.int8).max)
        ]:
            df[i] = df.loc[~df['dtype'].isin(['String', 'Categorical', 'Datetime'])].apply(
                lambda x: (x['min'] >= mn) and (x['max'] <= mx), axis=1
            )
    return df

def merge_type_df(dfs):
    """
    merge type data frame
    Parameters:
        dfs: list
            merge the type information of DataFrames
    Returns:
        df - return DataFrame
    Examples:
        >>> merge_type_df([pl.scan_csv(i).pipe(get_type_df) for i in glob(*.csv)])
    """
    return pd.concat(dfs, axis=0).pipe(lambda x: x.groupby(x.index).agg(
        {'min': 'min', 'max': 'max', 'na': 'sum', 'count': 'sum', 'n_unique': 'mean', 'dtype': 'max', 'f32': 'all', 'i32': 'all', 'i16': 'all', 'i8': 'all'})
    )

def get_type_pl(df_type, predefine={}, f32=True, i64=False, cat_max=np.inf, txt_cols = []):
    """
    get datatype for Polars DataFrame
    Parameters:
        df_type: pd.DataFrame
            the DataFrame whichi contains type information
        predefine: dict
            the predefined data type
        f32: Boolean
            Use f32 if possible
        i64: Boolean
            use i64 for all integer types
        cat_max: Int
            maximum number of categories for categorical types
        txt_cols: list
            Text type columns
    Returns:
        dict
            the dictionary of type mapping for pl.DataFrame loading
    Examples:
        >>> df_type = pl.scan_csv('test.csv').pipe(get_type_df)
        >>> dt = get_type_pl(df_type, {'id': pl.Int64})
    """
    ret_type = predefine.copy()
    
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Date')].index:
        ret_type = pl.Datetime
    if f32:
        for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int') & (df_type['na'] > 0) & (df_type['f32'])].index:
            ret_type[i] = pl.Float32
        for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Float') & (df_type['f32'])].index:
            ret_type[i] = pl.Float32
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int') & (df_type['na'] > 0)].index:
        ret_type[i] = pl.Float64
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Float')].index:
        ret_type[i] = pl.Float64
    if not i64:
        for col, ty in [('i8', pl.Int8), ('i16', pl.Int16), ('i32', pl.Int32)]:
            for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int') & df_type[col]].index:
                ret_type[i] = ty
        for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Float') & df_type[col]].index:
            ret_type[i] = ty
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int')].index:
        ret_type[i] = pl.Int64
    for i in df_type.loc[~df_type.index.isin(ret_type) & (df_type['n_unique'] <= cat_max) & (~df_type.index.isin(txt_cols))].index:
        ret_type[i] = pl.Categorical
    for i in df_type.loc[~df_type.index.isin(ret_type)].index:
        ret_type[i] = pl.String
    return ret_type

def get_type_pd(df_type, predefine={}, f32=True, i64=False, cat_max=np.inf, txt_cols = []):
    """
    get data type for pandas DataFrame
    Parameters:
        df_type: pd.DataFrame
            the DataFrame whichi contains type information
        predefine: dict
            the predefined data type
        f32: Boolean
            Use f32 if possible
        i64: Boolean
            use i64 for all integer types
        cat_max: Int
            maximum category number for categorical types
        txt_cols: list
            Text type columns
    Examples:
        >>> df_type = pl.scan_csv('test.csv').pipe(get_type_df)
        >>> dt = get_type_pd(df_type, {'id': 'int64'})
    """
    ret_type = predefine.copy()
    
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Date')].index:
        ret_type = 'datetime'
    if f32:
        for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int') & (df_type['na'] > 0) & (df_type['f32'])].index:
            ret_type[i] = 'float32'
        for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Float') & (df_type['f32'])].index:
            ret_type[i] = 'float32'
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int') & (df_type['na'] > 0)].index:
        ret_type[i] = 'float32'
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Float')].index:
        ret_type[i] = 'float32'
    if not i64:
        for col, ty in [('i8', 'int8'), ('i16', 'int16'), ('i32', 'int32')]:
            for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int') & df_type[col]].index:
                ret_type[i] = ty
        for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Float') & df_type[col]].index:
            ret_type[i] = ty
    for i in df_type.loc[~df_type.index.isin(ret_type) & df_type['dtype'].str.startswith('Int')].index:
        ret_type[i] = 'int64'
    for i in df_type.loc[~df_type.index.isin(ret_type) & (df_type['n_unique'] <= cat_max) & (~df_type.index.isin(txt_cols))].index:
        ret_type[i] = 'category'
    for i in df_type.loc[~df_type.index.isin(ret_type)].index:
        ret_type[i] = 'string'
    return ret_type

def join_and_assign(df1, df2):
    """
    Joins columns from the first DataFrame to the second DataFrame if they do not already exist in the second DataFrame.

    Args:
        df1 (pd.DataFrame): The source DataFrame containing columns to merge.
        df2 (pd.DataFrame): The target DataFrame to which columns from `df1` will be joined if not present.

    Returns:
        pd.DataFrame: The resulting DataFrame with `df1`'s columns added to `df2` where they were missing.
    """
    to_merge = [i for i in df1.columns if i not in df2.columns]
    if len(to_merge) == 0: return df2.copy()
    return df1[to_merge].join(df2)

def combine_cat(df, delimiter=''):
    """
    Combines multiple categorical columns in a DataFrame into a single categorical variable, in efficient way. 

    Parameters:
        df (pd.DataFrame): DataFrame where each column is of categorical dtype.
        delimiter (str): Delimiter

    Returns:
        pd.Series: A Series containing a new categorical variable that represents 
                   the unique combination of all input categorical columns.
    """
    return pd.Series(
        pd.Categorical.from_codes(
            df.apply(lambda x: x.cat.codes).dot(df.nunique().shift(1).fillna(1).astype('int').cumprod()), 
            [delimiter.join(i[::-1]) for i in product(*df[df.columns[::-1]].apply(lambda x: x.cat.categories.astype('str').tolist(), result_type='reduce'))]
        ), index=df.index
    )

def replace_cat(s, rule):
    """
    Replaces the categories in a pandas Categorical Series based on a given rule.

    Args:
        s (pd.Series): A pandas Series with categorical dtype (i.e., `pd.Categorical`).
        rule (Union[Dict[str, str], Callable[[str], str]]): A mapping or function that defines the 
            new category replacements.
            - If `rule` is a dictionary, the keys are the original categories and the values are the 
              replacements.
            - If `rule` is a function, it takes an original category and returns the new category.

    Returns:
        pd.Series: A new pandas Categorical object with the categories replaced 
        according to the given rule.

    Example:
        >>> s = pd.Series(['a', 'b', 'c'], dtype='category')
        >>> rule = {'a': 'x', 'b': 'x'}
        >>> replace_cat(s, rule)
        [x, x, c]
        Categories (2, object): [x, c]

        Or using a function:

        >>> rule = lambda x: x.upper()
        >>> replace_cat(s, rule)
        [A, B, C]
        Categories (3, object): [A, B, C]

    """
    code_replace = {}
    d = {}
    for c, n in zip(
        range(len(s.cat.categories)), 
        s.cat.categories
    ):
        new_cat = rule.get(n, n) if type(rule) == dict else rule(n)
        if new_cat in code_replace:
            d[c] = code_replace[new_cat]
        else:
            d[c] = len(code_replace)
            code_replace[new_cat] = len(code_replace)
    if s.isna().sum() > 0:
        return pd.Series(s.loc[s.notna()].pipe(
                lambda x: pd.Series(
                    pd.Categorical.from_codes(
                        x.cat.codes.map(d), list(code_replace.keys()), ordered=x.cat.ordered
                    ), index = x.index
                )
            ), index=s.index)
    else:
        return pd.Series(
            pd.Categorical.from_codes(
                s.cat.codes.map(d), list(code_replace.keys()), ordered=s.cat.ordered
            ), index=s.index
        )

def rearrange_cat(s_cat, cat_type, repl_rule, use_set=False):
    """
    Rearranges the categories of a pandas Categorical series based on a provided category type
    and a replacement rule for missing categories.

    Args:
        s_cat (pd.Series): A pandas Series of categorical values that need to be rearranged.
        cat_type (pd.api.types.CategoricalDtype): The target CategoricalDtype defining the desired
            order and structure of categories.
        repl_rule (callable): A function that defines a rule for handling categories in `s_cat`
            that are not found in `cat_type`. The function should take two arguments:
            `cat_vals` (the array of category values from `cat_type`) and `x` (a missing category
            from `s_cat`). It returns the value to replace the missing category with.
        use_set(Boolean): provide cat_vals with set
    Returns:
        pd.Categorical: A new pandas Categorical object with the categories of `s_cat` rearranged
        according to `cat_type`, and with missing categories replaced based on `repl_rule`.

    Example:
        >>> cat_type = pd.api.types.CategoricalDtype(categories=["a", "b", "c"])
        >>> s_cat = pd.Series(pd.Categorical(["a", "d", "b"]))
        >>> def repl_rule(cat_vals, x):
        ...     return 0  # Default to the first category if not found
        >>> rearrange_cat(s_cat, cat_type, repl_rule)
        [a, a, b]
        Categories (3, object): [a, b, c]
    """
    cat_vals = cat_type.categories.values
    s_map = pd.Series(np.arange(len(cat_vals)), cat_vals)
    if use_set:
        cat_vals_s = set(cat_vals)
        s_cat_map = pd.Series(s_cat.cat.categories.values, s_cat.cat.categories.values).apply(
            lambda x: s_map[x] if x in s_map else repl_rule(cat_vals_s, x)
        )
    else:
        s_cat_map = pd.Series(s_cat.cat.categories.values, s_cat.cat.categories.values).apply(
            lambda x: s_map[x] if x in s_map else repl_rule(cat_vals, x)
        )
    notna = s_cat.notna()
    return s_cat.loc[notna].pipe(
        lambda x: pd.Series(pd.Series(pd.Categorical.from_codes(x.map(s_cat_map), cat_vals), index=x.index), index=s_cat.index)
    ) if notna.sum() != len(s_cat) else pd.Series(pd.Categorical.from_codes(s_cat.map(s_cat_map), cat_vals), index=s_cat.index)

def split_preprocessor_var(s_names, org_names):
    return s_names.str.split('__|_').apply(
        lambda x: (x, [i for i in range(len(x), 0, -1) if '_'.join(x[1:i]) in org_names][0])
    ).apply(
        lambda x: pd.Series(['_'.join(x[0][:x[1]]), ''.join(x[0][x[1]:])], index=['var1', 'var2'])
    )

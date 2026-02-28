import pytest
import numpy as np
import pandas as pd

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    tf = None
    HAS_TF = False

requires_tf = pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
requires_polars = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def pd_df():
    np.random.seed(0)
    n = 200
    return pd.DataFrame({
        'color':  pd.Categorical(np.random.choice(['red', 'green', 'blue'], n)),
        'grade':  pd.Categorical(np.random.choice(['A', 'B', 'C', 'D'], n)),
        'num1':   np.random.randn(n).astype(np.float32),
        'num2':   np.random.randn(n).astype(np.float32),
    })


@pytest.fixture
def clf_target():
    np.random.seed(0)
    return np.random.choice(['cat', 'dog', 'bird'], 200)


@pytest.fixture
def bin_target():
    np.random.seed(0)
    return np.random.choice([0, 1], 200)


@pytest.fixture
def reg_target():
    np.random.seed(0)
    return np.random.randn(200).astype(np.float32)


# ======================================================================
# _make_tf_dataset
# ======================================================================

class TestMakeTfDataset:

    @requires_tf
    def test_num_pandas(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset
        specs = [('feats', ['num1', 'num2'], 'num')]
        ds, model = _make_tf_dataset(pd_df, specs)
        batch = next(iter(ds.batch(10)))
        assert batch['feats'].dtype == tf.float32
        assert batch['feats'].shape == (10, 2)

    @requires_tf
    def test_int_pandas(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset
        df = pd_df.copy()
        df['code'] = np.arange(len(df))
        specs = [('code', ['code'], 'int')]
        ds, model = _make_tf_dataset(df, specs)
        batch = next(iter(ds.batch(10)))
        assert batch['code'].dtype == tf.int32

    @requires_tf
    def test_str_pandas(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset
        specs = [('color', ['color'], 'str')]
        ds, model = _make_tf_dataset(pd_df, specs)
        batch = next(iter(ds.batch(10)))
        assert batch['color'].dtype == tf.string

    @requires_tf
    def test_cat_dtype_as_str(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset
        specs = [('color', ['color'], 'str')]
        ds, model = _make_tf_dataset(pd_df, specs)
        batch = next(iter(ds.batch(5)))
        assert batch['color'].shape == (5, 1)

    @requires_tf
    def test_returns_model(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset, _DatasetInputModel
        specs = [('feats', ['num1', 'num2'], 'num')]
        ds, model = _make_tf_dataset(pd_df, specs)
        assert isinstance(model, _DatasetInputModel)
        # cont passthrough: output same as input
        inp = tf.keras.Input(shape=(2,), name='feats')
        out = model({'feats': inp})
        assert out['feats'].shape[-1] == 2

    @requires_tf
    def test_embedding_model(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset, _DatasetInputModel
        specs = [
            ('color', ['color'], ('Embedding', 4, 'str')),
            ('__cont__', ['num1', 'num2'], 'num'),
        ]
        ds, model = _make_tf_dataset(pd_df, specs)
        assert isinstance(model, _DatasetInputModel)
        # cat output is embedding dim, cont is passthrough
        cat_inp  = tf.keras.Input(shape=(1,), dtype='string', name='color')
        cont_inp = tf.keras.Input(shape=(2,), name='__cont__')
        out = model({'color': cat_inp, '__cont__': cont_inp})
        assert out['color'].shape[-1] == 4
        assert out['__cont__'].shape[-1] == 2

    @requires_tf
    @pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
    def test_polars(self):
        from mllabs.nn._input import _make_tf_dataset
        df = pl.DataFrame({
            'a': np.random.randn(50).astype(np.float32),
            'b': np.random.randn(50).astype(np.float32),
        })
        ds, _ = _make_tf_dataset(df, [('feats', ['a', 'b'], 'num')])
        batch = next(iter(ds.batch(10)))
        assert batch['feats'].shape == (10, 2)

    @requires_tf
    def test_numpy(self):
        from mllabs.nn._input import _make_tf_dataset
        arr = np.random.randn(50, 3).astype(np.float32)
        ds, _ = _make_tf_dataset(arr, [('feats', [0, 1, 2], 'num')])
        batch = next(iter(ds.batch(10)))
        assert batch['feats'].shape == (10, 3)

    def test_unknown_type_raises(self, pd_df):
        from mllabs.nn._input import _make_tf_dataset
        with pytest.raises(ValueError):
            _make_tf_dataset(pd_df, [('x', ['num1'], 'unknown')])


# ======================================================================
# _analyze_cols / build_embedding_models
# ======================================================================

class TestEmbeddingEncoder:

    def test_analyze_string_col(self, pd_df):
        from mllabs.nn._input import _analyze_cols
        col_info = _analyze_cols(pd_df, ['color', 'grade'])
        assert col_info['color']['type'] == 'string'
        assert col_info['grade']['type'] == 'string'
        assert col_info['color']['cardinality'] == 3
        assert col_info['grade']['cardinality'] == 4

    def test_analyze_ordinal_int(self):
        from mllabs.nn._input import _analyze_cols
        df = pd.DataFrame({'x': [0, 1, 2, 3, 0, 1]})
        col_info = _analyze_cols(df, ['x'])
        assert col_info['x']['type'] == 'ordinal_int'
        assert col_info['x']['cardinality'] == 4

    def test_analyze_non_ordinal_int(self):
        from mllabs.nn._input import _analyze_cols
        df = pd.DataFrame({'x': [0, 5, 10, 0, 5]})
        col_info = _analyze_cols(df, ['x'])
        assert col_info['x']['type'] == 'int'

    @requires_tf
    def test_build_returns_dict_of_models(self, pd_df):
        from mllabs.nn._input import build_embedding_models
        models = build_embedding_models(pd_df, ['color', 'grade'], {'color': 4, 'grade': 3})
        assert set(models.keys()) == {'color', 'grade'}
        # each value is a keras Sequential
        for col, model in models.items():
            assert isinstance(model, tf.keras.Sequential)

    @requires_tf
    def test_build_output_shape(self, pd_df):
        from mllabs.nn._input import build_embedding_models, _analyze_cols
        models = build_embedding_models(pd_df, ['color', 'grade'], {'color': 4, 'grade': 3})
        col_info = _analyze_cols(pd_df, ['color', 'grade'])
        for col, dim in [('color', 4), ('grade', 3)]:
            t = col_info[col]['type']
            dtype = 'int32' if t in ('ordinal_int', 'int') else 'string'
            inp = tf.keras.Input(shape=(), dtype=dtype, name=col)
            out = models[col](inp)
            assert out.shape[-1] == dim

    @requires_tf
    def test_build_auto_dim(self, pd_df):
        from mllabs.nn._input import build_embedding_models, _analyze_cols, _auto_emb_dim
        models = build_embedding_models(pd_df, ['color'])
        col_info = _analyze_cols(pd_df, ['color'])
        inp = tf.keras.Input(shape=(), dtype='string', name='color')
        out = models['color'](inp)
        expected_dim = _auto_emb_dim(col_info['color']['cardinality'])
        assert out.shape[-1] == expected_dim


# ======================================================================
# SimpleConcatHead
# ======================================================================

class TestSimpleConcatHead:

    @requires_tf
    def test_concat_emb_and_cont(self):
        from mllabs.nn._head import SimpleConcatHead

        inp1 = tf.keras.Input(shape=(), dtype='string', name='color')
        inp2 = tf.keras.Input(shape=(), dtype='string', name='grade')
        emb1 = tf.keras.layers.Embedding(4, 4, name='emb_color')(
            tf.keras.layers.StringLookup(vocabulary=['red', 'green', 'blue'], mask_token=None)(inp1)
        )
        emb2 = tf.keras.layers.Embedding(5, 3, name='emb_grade')(
            tf.keras.layers.StringLookup(vocabulary=['A', 'B', 'C', 'D'], mask_token=None)(inp2)
        )
        cont_input = tf.keras.Input(shape=(2,), name='__cont__')
        head = SimpleConcatHead()
        out = head.build([inp1, inp2], [emb1, emb2], cont_input)
        # (4 + 3 + 2) = 9
        assert out.shape[-1] == 9

    @requires_tf
    def test_no_cont(self):
        from mllabs.nn._head import SimpleConcatHead

        inp1 = tf.keras.Input(shape=(), dtype='int32', name='c1')
        inp2 = tf.keras.Input(shape=(), dtype='int32', name='c2')
        emb1 = tf.keras.layers.Embedding(5, 4)(inp1)
        emb2 = tf.keras.layers.Embedding(5, 3)(inp2)
        head = SimpleConcatHead()
        out = head.build([inp1, inp2], [emb1, emb2], None)
        assert out.shape[-1] == 7

    @requires_tf
    def test_emb_dropout(self):
        from mllabs.nn._head import SimpleConcatHead

        inp = tf.keras.Input(shape=(), dtype='int32', name='c')
        emb = tf.keras.layers.Embedding(5, 4)(inp)
        head = SimpleConcatHead(emb_dropout=0.2)
        out = head.build([inp], [emb], None)
        assert out.shape[-1] == 4


# ======================================================================
# DenseBody
# ======================================================================

class TestDenseBody:

    @requires_tf
    def test_basic(self):
        from mllabs.nn._body import DenseBody
        inp = tf.keras.Input(shape=(16,))
        body = DenseBody(layers=(32, 16))
        out = body.build(inp)
        assert out.shape[-1] == 16

    @requires_tf
    def test_batch_norm(self):
        from mllabs.nn._body import DenseBody
        inp = tf.keras.Input(shape=(16,))
        body = DenseBody(layers=(32,), batch_norm=True)
        out = body.build(inp)
        assert out.shape[-1] == 32

    @requires_tf
    def test_no_dropout(self):
        from mllabs.nn._body import DenseBody
        inp = tf.keras.Input(shape=(8,))
        body = DenseBody(layers=(16,), dropout=0.0)
        out = body.build(inp)
        assert out.shape[-1] == 16


# ======================================================================
# Tails
# ======================================================================

class TestTails:

    @requires_tf
    def test_logit_tail(self):
        from mllabs.nn._tail import LogitTail
        inp = tf.keras.Input(shape=(32,))
        tail = LogitTail()
        out = tail.build(inp, n_output=3)
        assert out.shape[-1] == 3
        assert tail.loss() == 'sparse_categorical_crossentropy'
        assert 'accuracy' in tail.compile_metrics()

    @requires_tf
    def test_binary_logit_tail(self):
        from mllabs.nn._tail import BinaryLogitTail
        inp = tf.keras.Input(shape=(32,))
        tail = BinaryLogitTail()
        out = tail.build(inp, n_output=1)
        assert out.shape[-1] == 1
        assert tail.loss() == 'binary_crossentropy'

    @requires_tf
    def test_regression_tail(self):
        from mllabs.nn._tail import RegressionTail
        inp = tf.keras.Input(shape=(32,))
        tail = RegressionTail()
        out = tail.build(inp, n_output=1)
        assert out.shape[-1] == 1
        assert tail.loss() == 'mse'
        assert 'mae' in tail.compile_metrics()


# ======================================================================
# NNClassifier
# ======================================================================

class TestNNClassifier:

    @requires_tf
    def test_multiclass_fit_predict(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping_patience=0)
        clf.fit(pd_df, clf_target)
        preds = clf.predict(pd_df)
        assert preds.shape == (200,)
        assert set(preds).issubset(set(clf_target))

    @requires_tf
    def test_binary_fit_predict(self, pd_df, bin_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping_patience=0)
        clf.fit(pd_df, bin_target)
        preds = clf.predict(pd_df)
        assert set(preds).issubset({0, 1})

    @requires_tf
    def test_predict_proba_shape(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping_patience=0)
        clf.fit(pd_df, clf_target)
        proba = clf.predict_proba(pd_df)
        assert proba.shape == (200, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @requires_tf
    def test_binary_predict_proba_shape(self, pd_df, bin_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping_patience=0)
        clf.fit(pd_df, bin_target)
        proba = clf.predict_proba(pd_df)
        assert proba.shape == (200, 2)

    @requires_tf
    def test_eval_set(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping_patience=0)
        clf.fit(pd_df, clf_target, eval_set=[(pd_df, clf_target)])
        assert 'valid' in clf.evals_result_
        assert 'train' in clf.evals_result_

    @requires_tf
    def test_validation_fraction(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.2,
                           early_stopping_patience=0)
        clf.fit(pd_df, clf_target)
        assert 'valid' in clf.evals_result_

    @requires_tf
    def test_early_stopping(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=50, batch_size=64, validation_fraction=0.2,
                           early_stopping_patience=2)
        clf.fit(pd_df, clf_target)
        # validation data is present → val_loss tracked
        assert 'valid' in clf.evals_result_
        assert isinstance(clf.best_epoch_, int)

    @requires_tf
    def test_evals_result_structure(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(epochs=3, batch_size=64, validation_fraction=0.2,
                           early_stopping_patience=0)
        clf.fit(pd_df, clf_target)
        assert 'loss' in clf.evals_result_['train']
        assert 'loss' in clf.evals_result_['valid']
        assert len(clf.evals_result_['train']['loss']) == 3

    @requires_tf
    def test_custom_body(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier, DenseBody
        clf = NNClassifier(
            body=DenseBody(layers=(32,), dropout=0.1),
            epochs=2, batch_size=64,
            validation_fraction=0.0, early_stopping_patience=0,
        )
        clf.fit(pd_df, clf_target)
        assert clf.predict(pd_df).shape == (200,)

    @requires_tf
    def test_no_cat_cols(self, reg_target):
        from mllabs.nn import NNClassifier
        df = pd.DataFrame({'a': np.random.randn(200), 'b': np.random.randn(200)})
        y = np.random.choice(['x', 'y'], 200)
        clf = NNClassifier(epochs=2, batch_size=64, validation_fraction=0.0,
                           early_stopping_patience=0)
        clf.fit(df, y)
        assert clf.predict(df).shape == (200,)

    @requires_tf
    def test_explicit_embedding_dims(self, pd_df, clf_target):
        from mllabs.nn import NNClassifier
        clf = NNClassifier(
            embedding_dims={'color': 8, 'grade': 2},
            epochs=2, batch_size=64,
            validation_fraction=0.0, early_stopping_patience=0,
        )
        clf.fit(pd_df, clf_target)
        assert clf.embedding_dims_['color'] == 8
        assert clf.embedding_dims_['grade'] == 2


# ======================================================================
# NNRegressor
# ======================================================================

class TestNNRegressor:

    @requires_tf
    def test_fit_predict(self, pd_df, reg_target):
        from mllabs.nn import NNRegressor
        reg = NNRegressor(epochs=2, batch_size=64, validation_fraction=0.0,
                          early_stopping_patience=0)
        reg.fit(pd_df, reg_target)
        preds = reg.predict(pd_df)
        assert preds.shape == (200,)
        assert preds.dtype == np.float32

    @requires_tf
    def test_eval_set(self, pd_df, reg_target):
        from mllabs.nn import NNRegressor
        reg = NNRegressor(epochs=2, batch_size=64, validation_fraction=0.0,
                          early_stopping_patience=0)
        reg.fit(pd_df, reg_target, eval_set=[(pd_df, reg_target)])
        assert 'valid' in reg.evals_result_
        assert 'mae' in reg.evals_result_['valid']

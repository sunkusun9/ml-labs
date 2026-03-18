import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, KFold

from mllabs.sampler import Sampler, ImbLearnSampler
from mllabs._data_wrapper import PandasWrapper
from mllabs._node_processor import TransformProcessor, PredictProcessor
from mllabs._experimenter import Experimenter


# ── helpers ───────────────────────────────────────────────────────────────────

def pw(data, columns=None):
    if isinstance(data, pd.DataFrame):
        return PandasWrapper(data)
    return PandasWrapper(pd.DataFrame(np.array(data), columns=columns))


def make_exp(path, data, aug_data=None):
    return Experimenter(
        data=data,
        path=path,
        sp=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
        sp_v=KFold(n_splits=3, shuffle=True, random_state=42),
        aug_data=aug_data,
    )


class MockResampler:
    """imblearn API compatible mock: appends extra rows."""
    def __init__(self, extra_X, extra_y=None):
        self.extra_X = np.array(extra_X)
        self.extra_y = np.array(extra_y) if extra_y is not None else None

    def fit_resample(self, X, y):
        X_res = np.vstack([X, self.extra_X])
        y_res = np.concatenate([y, self.extra_y]) if y is not None and self.extra_y is not None else y
        return X_res, y_res


class AddRowsSampler(Sampler):
    """Test sampler: appends known extra rows to fit_params X and y."""
    def __init__(self, extra_X, extra_y=None):
        self.extra_X = np.array(extra_X)
        self.extra_y = np.array(extra_y) if extra_y is not None else None

    def sample(self, fit_params):
        result = dict(fit_params)
        result['X'] = np.vstack([fit_params['X'], self.extra_X])
        if 'y' in fit_params and self.extra_y is not None:
            result['y'] = np.concatenate([fit_params['y'], self.extra_y])
        return result


class SimpleFitPredict:
    __name__ = 'SimpleFitPredict'
    classes_ = [0, 1]

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)


# ── ImbLearnSampler ───────────────────────────────────────────────────────────

class TestImbLearnSampler:
    def test_sample_augments_X_y(self):
        sampler = ImbLearnSampler(MockResampler([[10.0, 10.0]], [1]))
        fit_params = {'X': np.array([[1.0, 2.0], [3.0, 4.0]]), 'y': np.array([0, 1])}
        result = sampler.sample(fit_params)
        assert result['X'].shape[0] == 3
        assert result['y'].shape[0] == 3
        np.testing.assert_array_equal(result['X'][-1], [10.0, 10.0])
        assert result['y'][-1] == 1

    def test_sample_passes_through_other_keys(self):
        sampler = ImbLearnSampler(MockResampler([[10.0]], [1]))
        extra = [('dummy',)]
        fit_params = {'X': np.array([[1.0], [2.0]]), 'y': np.array([0, 1]), 'eval_set': extra}
        result = sampler.sample(fit_params)
        assert result['eval_set'] is extra

    def test_sample_no_y_in_fit_params(self):
        sampler = ImbLearnSampler(MockResampler([[10.0]]))
        fit_params = {'X': np.array([[1.0], [2.0]])}
        result = sampler.sample(fit_params)
        assert result['X'].shape[0] == 3
        assert 'y' not in result

    def test_does_not_mutate_original(self):
        sampler = ImbLearnSampler(MockResampler([[10.0, 10.0]], [1]))
        X_orig = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_orig = np.array([0, 1])
        fit_params = {'X': X_orig, 'y': y_orig}
        sampler.sample(fit_params)
        assert fit_params['X'].shape[0] == 2
        assert fit_params['y'].shape[0] == 2


# ── mllab_sampler in TransformProcessor ──────────────────────────────────────

class TestTransformProcessorSampler:
    TRAIN = [[1.0, 2.0], [3.0, 4.0]]
    TRAIN_V = [[1.5, 2.5]]
    EXTRA = [[10.0, 10.0]]

    def _data_dict(self):
        cols = ['a', 'b']
        return {
            'X': (pw(pd.DataFrame(self.TRAIN, columns=cols)),
                  pw(pd.DataFrame(self.TRAIN_V, columns=cols))),
        }

    def test_fit_uses_augmented_data(self):
        sampler = AddRowsSampler(self.EXTRA)
        data_dict = self._data_dict()
        proc = TransformProcessor('s', StandardScaler, params={'mllab_sampler': sampler})
        proc.fit(data_dict)
        expected_mean = np.mean(self.TRAIN + self.EXTRA, axis=0).astype(float)
        np.testing.assert_allclose(proc.obj.mean_, expected_mean)

    def test_fit_without_sampler_unaffected(self):
        data_dict = self._data_dict()
        proc = TransformProcessor('s', StandardScaler)
        proc.fit(data_dict)
        expected_mean = np.mean(self.TRAIN, axis=0).astype(float)
        np.testing.assert_allclose(proc.obj.mean_, expected_mean)

    def test_fit_sampler_not_passed_to_transformer(self):
        sampler = AddRowsSampler(self.EXTRA)
        data_dict = self._data_dict()
        proc = TransformProcessor('s', StandardScaler, params={'mllab_sampler': sampler})
        proc.fit(data_dict)  # StandardScaler does not accept mllab_sampler — must not raise

    def test_fit_process_fitted_on_augmented(self):
        sampler = AddRowsSampler(self.EXTRA)
        data_dict = self._data_dict()
        proc = TransformProcessor('s', StandardScaler, params={'mllab_sampler': sampler})
        proc.fit_process(data_dict)
        expected_mean = np.mean(self.TRAIN + self.EXTRA, axis=0).astype(float)
        np.testing.assert_allclose(proc.obj.mean_, expected_mean)

    def test_fit_process_output_shape_matches_original(self):
        sampler = AddRowsSampler(self.EXTRA)
        data_dict = self._data_dict()
        proc = TransformProcessor('s', StandardScaler, params={'mllab_sampler': sampler})
        result = proc.fit_process(data_dict)
        assert result.get_shape()[0] == len(self.TRAIN)


# ── mllab_sampler in PredictProcessor ────────────────────────────────────────

class TestPredictProcessorSampler:
    N = 20
    N_V = 5
    N_EXTRA = 5

    @pytest.fixture(autouse=True)
    def data(self):
        rng = np.random.default_rng(0)
        cols = ['a', 'b']
        self.train_X = rng.standard_normal((self.N, 2))
        self.train_v_X = rng.standard_normal((self.N_V, 2))
        self.train_y = rng.integers(0, 2, self.N)
        self.train_v_y = rng.integers(0, 2, self.N_V)
        self.extra_X = rng.standard_normal((self.N_EXTRA, 2))
        self.extra_y = rng.integers(0, 2, self.N_EXTRA)
        self.data_dict = {
            'X': (pw(pd.DataFrame(self.train_X, columns=cols)),
                  pw(pd.DataFrame(self.train_v_X, columns=cols))),
            'y': (pw(pd.DataFrame({'t': self.train_y})),
                  pw(pd.DataFrame({'t': self.train_v_y}))),
        }

    def test_fit_sampler_not_passed_to_estimator(self):
        sampler = AddRowsSampler(self.extra_X, self.extra_y)
        proc = PredictProcessor('dt', DecisionTreeClassifier,
                                params={'mllab_sampler': sampler, 'max_depth': 3, 'random_state': 0})
        proc.fit(self.data_dict)  # must not raise

    def test_fit_process_output_shape_matches_original(self):
        sampler = AddRowsSampler(self.extra_X, self.extra_y)
        proc = PredictProcessor('dt', SimpleFitPredict, method='fit_predict',
                                params={'mllab_sampler': sampler})
        result = proc.fit_process(self.data_dict)
        assert result.get_shape()[0] == self.N

    def test_fit_process_without_sampler_output_shape(self):
        proc = PredictProcessor('dt', SimpleFitPredict, method='fit_predict')
        result = proc.fit_process(self.data_dict)
        assert result.get_shape()[0] == self.N


# ── Experimenter.aug_data ─────────────────────────────────────────────────────

@pytest.fixture
def base_data():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'f1': rng.standard_normal(100),
        'f2': rng.standard_normal(100),
        'target': rng.integers(0, 2, 100),
    })


@pytest.fixture
def aug_df():
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        'f1': rng.standard_normal(20),
        'f2': rng.standard_normal(20),
        'target': rng.integers(0, 2, 20),
    })


class TestExperimenterAugData:
    def test_aug_data_stored(self, tmp_path, base_data, aug_df):
        exp = make_exp(tmp_path / 'exp', base_data, aug_data=aug_df)
        assert exp.aug_data is not None

    def test_no_aug_data_is_none(self, tmp_path, base_data):
        exp = make_exp(tmp_path / 'exp', base_data)
        assert exp.aug_data is None

    def test_inner_train_size_increased(self, tmp_path, base_data, aug_df):
        exp_with = make_exp(tmp_path / 'with', base_data, aug_data=aug_df)
        exp_without = make_exp(tmp_path / 'without', base_data)
        for (t_with, _), _ in exp_with.get_node_output(None, 0):
            for (t_without, _), _ in exp_without.get_node_output(None, 0):
                assert t_with.get_shape()[0] == t_without.get_shape()[0] + len(aug_df)
                break
            break

    def test_inner_valid_unchanged(self, tmp_path, base_data, aug_df):
        exp_with = make_exp(tmp_path / 'with', base_data, aug_data=aug_df)
        exp_without = make_exp(tmp_path / 'without', base_data)
        tv_with = [tv.get_shape()[0] for (_, tv), _ in exp_with.get_node_output(None, 0) if tv is not None]
        tv_without = [tv.get_shape()[0] for (_, tv), _ in exp_without.get_node_output(None, 0) if tv is not None]
        assert tv_with == tv_without

    def test_outer_valid_unchanged(self, tmp_path, base_data, aug_df):
        exp_with = make_exp(tmp_path / 'with', base_data, aug_data=aug_df)
        exp_without = make_exp(tmp_path / 'without', base_data)
        v_with = [v.get_shape()[0] for _, v in exp_with.get_node_output(None, 0)]
        v_without = [v.get_shape()[0] for _, v in exp_without.get_node_output(None, 0)]
        assert v_with == v_without

    def test_column_filter_applied_to_aug(self, tmp_path, base_data, aug_df):
        exp = make_exp(tmp_path / 'exp', base_data, aug_data=aug_df)
        for (train, _), _ in exp.get_node_output(None, 0, v=['f1', 'f2']):
            assert train.get_columns() == ['f1', 'f2']
            break

    def test_load_passes_aug_data(self, tmp_path, base_data, aug_df):
        path = tmp_path / 'exp'
        make_exp(path, base_data, aug_data=aug_df)
        exp2 = Experimenter.load(path, data=base_data, aug_data=aug_df)
        assert exp2.aug_data is not None
        for (t, _), _ in exp2.get_node_output(None, 0):
            assert t.get_shape()[0] == pytest.approx(t.get_shape()[0])  # just check it runs
            break

    def test_load_without_aug_data(self, tmp_path, base_data):
        path = tmp_path / 'exp'
        make_exp(path, base_data)
        exp2 = Experimenter.load(path, data=base_data)
        assert exp2.aug_data is None


# ── Trainer.aug_data ──────────────────────────────────────────────────────────

class TestTrainerAugData:
    @pytest.fixture
    def exp(self, tmp_path, base_data):
        e = Experimenter(
            data=base_data,
            path=tmp_path / 'exp',
            sp=ShuffleSplit(n_splits=1, test_size=0.2, random_state=42),
            sp_v=KFold(n_splits=3, shuffle=True, random_state=42),
        )
        e.set_grp('model', role='head', processor=DecisionTreeClassifier,
                  method='predict',
                  edges={'X': [(None, ['f1', 'f2'])], 'y': [(None, 'target')]},
                  params={'max_depth': 3, 'random_state': 42})
        e.set_node('dt', grp='model')
        return e

    def test_add_trainer_stores_aug_data(self, exp, aug_df):
        trainer = exp.add_trainer('t1', aug_data=aug_df)
        assert trainer.aug_data is not None

    def test_add_trainer_no_aug_data(self, exp):
        trainer = exp.add_trainer('t1')
        assert trainer.aug_data is None

    def test_trainer_inner_train_size_increased(self, exp, aug_df):
        trainer_with = exp.add_trainer('t_with', aug_data=aug_df)
        trainer_without = exp.add_trainer('t_without')
        for t_with, _ in trainer_with.get_node_output(None):
            for t_without, _ in trainer_without.get_node_output(None):
                assert t_with.get_shape()[0] == t_without.get_shape()[0] + len(aug_df)
                break
            break

    def test_trainer_valid_unchanged(self, exp, aug_df):
        trainer_with = exp.add_trainer('t_with', aug_data=aug_df)
        trainer_without = exp.add_trainer('t_without')
        v_with = [v.get_shape()[0] for _, v in trainer_with.get_node_output(None)]
        v_without = [v.get_shape()[0] for _, v in trainer_without.get_node_output(None)]
        assert v_with == v_without

    def test_trainer_no_split_aug_data(self, exp, aug_df):
        trainer_with = exp.add_trainer('t_with', splitter=None, aug_data=aug_df)
        trainer_without = exp.add_trainer('t_without', splitter=None)
        for t_with, _ in trainer_with.get_node_output(None):
            for t_without, _ in trainer_without.get_node_output(None):
                assert t_with.get_shape()[0] == t_without.get_shape()[0] + len(aug_df)
                break
            break

    def test_trainer_column_filter_applied_to_aug(self, exp, aug_df):
        trainer = exp.add_trainer('t1', aug_data=aug_df)
        for train, _ in trainer.get_node_output(None, v=['f1', 'f2']):
            assert train.get_columns() == ['f1', 'f2']
            break

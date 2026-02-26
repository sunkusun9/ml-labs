import pytest
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, KFold

from mllabs._experimenter import Experimenter
from mllabs._inferencer import Inferencer
from mllabs._data_wrapper import unwrap


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'f1': np.random.randn(n),
        'f2': np.random.randn(n),
        'f3': np.random.randn(n),
        'target': np.random.randint(0, 2, n),
    })


@pytest.fixture
def exp(tmp_path, sample_data):
    e = Experimenter(
        data=sample_data,
        path=tmp_path / 'exp',
        sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        sp_v=KFold(n_splits=3, shuffle=True, random_state=42),
    )
    e.set_grp('scale', role='stage', processor=StandardScaler,
              method='transform', edges={'X': [(None, ['f1', 'f2', 'f3'])]})
    e.set_node('scaler', grp='scale')
    e.set_grp('model', role='head', processor=DecisionTreeClassifier,
              method='predict',
              edges={'X': [('scaler', None)], 'y': [(None, 'target')]},
              params={'max_depth': 3, 'random_state': 42})
    e.set_node('dt', grp='model')
    e.build()
    e.exp()
    return e


@pytest.fixture
def trained_trainer(exp):
    trainer = exp.add_trainer('t1')
    trainer.select_head(['dt'])
    trainer.train()
    return trainer


class TestToInferencer:
    def test_basic_creation(self, trained_trainer):
        inf = trained_trainer.to_inferencer()
        assert isinstance(inf, Inferencer)
        assert inf.n_splits == trained_trainer.get_n_splits()
        assert inf.selected_stages == trained_trainer.selected_stages
        assert inf.selected_heads == trained_trainer.selected_heads

    def test_node_objs_are_processor_lists(self, trained_trainer):
        inf = trained_trainer.to_inferencer()
        for name, objs in inf.node_objs.items():
            assert isinstance(objs, list)
            assert len(objs) == inf.n_splits
            assert hasattr(objs[0], 'process')

    def test_minimal_pipeline(self, trained_trainer):
        inf = trained_trainer.to_inferencer()
        assert 'scaler' in inf.pipeline.nodes
        assert 'dt' in inf.pipeline.nodes

    def test_not_trained_raises(self, exp):
        trainer = exp.add_trainer('t_no_train')
        trainer.select_head(['dt'])
        with pytest.raises(RuntimeError, match="not built"):
            trainer.to_inferencer()

    def test_v_stored(self, trained_trainer):
        inf = trained_trainer.to_inferencer(v=[0])
        assert inf.v == [0]


class TestProcess:
    def test_mean_agg(self, trained_trainer, sample_data):
        inf = trained_trainer.to_inferencer()
        result = inf.process(sample_data, agg='mean')
        assert result.shape[0] == len(sample_data)

    def test_mode_agg(self, trained_trainer, sample_data):
        inf = trained_trainer.to_inferencer()
        result = inf.process(sample_data, agg='mode')
        assert result.shape[0] == len(sample_data)

    def test_callable_agg(self, trained_trainer, sample_data):
        inf = trained_trainer.to_inferencer()
        result = inf.process(sample_data, agg=lambda results: results[0])
        assert result.shape[0] == len(sample_data)

    def test_none_agg(self, trained_trainer, sample_data):
        inf = trained_trainer.to_inferencer()
        results = inf.process(sample_data, agg=None)
        assert isinstance(results, list)
        assert len(results) == inf.n_splits

    def test_v_parameter(self, exp, sample_data):
        exp.set_grp('model_proba', role='head', processor=DecisionTreeClassifier,
                    method='predict_proba',
                    edges={'X': [('scaler', None)], 'y': [(None, 'target')]},
                    params={'max_depth': 3, 'random_state': 42})
        exp.set_node('dt_proba', grp='model_proba')
        exp.build()
        exp.exp()
        trainer = exp.add_trainer('t_proba')
        trainer.select_head(['dt_proba'])
        trainer.train()
        inf = trainer.to_inferencer(v=slice(-1, None))
        result = inf.process(sample_data)
        assert result.shape[1] == 1

    def test_single_split(self, exp, sample_data):
        trainer = exp.add_trainer('t_nosplit', splitter=None)
        trainer.select_head(['dt'])
        trainer.train()
        inf = trainer.to_inferencer()
        result = inf.process(sample_data)
        assert result.shape[0] == len(sample_data)

    def test_unknown_agg_raises(self, trained_trainer, sample_data):
        inf = trained_trainer.to_inferencer()
        with pytest.raises(ValueError, match="Unknown agg"):
            inf.process(sample_data, agg='unknown')


class TestSaveLoad:
    def test_save_load_roundtrip(self, trained_trainer, tmp_path):
        inf = trained_trainer.to_inferencer()
        save_path = tmp_path / 'inferencer'
        inf.save(save_path)

        loaded = Inferencer.load(save_path)
        assert loaded.n_splits == inf.n_splits
        assert loaded.selected_stages == inf.selected_stages
        assert loaded.selected_heads == inf.selected_heads
        assert set(loaded.node_objs.keys()) == set(inf.node_objs.keys())

    def test_loaded_process_matches(self, trained_trainer, sample_data, tmp_path):
        inf = trained_trainer.to_inferencer()
        save_path = tmp_path / 'inferencer'
        inf.save(save_path)

        loaded = Inferencer.load(save_path)
        original = inf.process(sample_data, agg=None)
        loaded_result = loaded.process(sample_data, agg=None)

        assert len(original) == len(loaded_result)
        for orig, load in zip(original, loaded_result):
            np.testing.assert_array_equal(unwrap(orig), unwrap(load))

    def test_save_creates_file(self, trained_trainer, tmp_path):
        inf = trained_trainer.to_inferencer()
        save_path = tmp_path / 'inferencer'
        inf.save(save_path)
        assert (save_path / '__inferencer.pkl').exists()

    def test_save_load_with_v(self, trained_trainer, sample_data, tmp_path):
        exp = trained_trainer
        inf = trained_trainer.to_inferencer(v=[0])
        save_path = tmp_path / 'inferencer_v'
        inf.save(save_path)

        loaded = Inferencer.load(save_path)
        assert loaded.v == [0]

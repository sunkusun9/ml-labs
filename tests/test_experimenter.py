import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, KFold

from mllabs._experimenter import Experimenter, DataCache
from mllabs._store import NodeStore
from mllabs._flow import DataFlow
from mllabs._pipeline import Pipeline
from mllabs import Connector, MetricCollector


class BadProcessor:
    __name__ = 'BadProcessor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        raise ValueError("intentional error")
    def transform(self, X):
        pass


class ErrorProcessor:
    __name__ = 'ErrorProcessor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        raise TypeError("test error msg")
    def transform(self, X):
        pass


class BadPredictor:
    __name__ = 'BadPredictor'
    def __init__(self, **kwargs):
        pass
    def fit(self, X, y=None):
        raise RuntimeError("predict error")
    def predict(self, X):
        pass


def accuracy_metric(y, pred):
    return (y.values == pred.values).mean()

def dummy_metric(y, pred):
    return 0.5


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
    )
    return e


def _setup_stage(e):
    e.set_grp('scale', role='stage', processor=StandardScaler,
              method='transform', edges={'X': [(None, ['f1', 'f2', 'f3'])]})
    e.set_node('scaler', grp='scale')


def _setup_head(e):
    e.set_grp('model', role='head', processor=DecisionTreeClassifier,
              method='predict', edges={'X': [(None, ['f1', 'f2', 'f3'])],
                                        'y': [(None, 'target')]},
              params={'max_depth': 3, 'random_state': 42})
    e.set_node('dt', grp='model')


def _setup_full(e):
    _setup_stage(e)
    _setup_head(e)


def _flow(exp, outer=0, inner=0):
    return exp.outer_folds[outer].train_data_flows[inner]

def _store(exp, outer=0, inner=0):
    return exp.outer_folds[outer].artifact_stores[inner]


class TestDataCache:
    def test_put_get(self):
        c = DataCache(maxsize=1024**3)
        data = np.array([1, 2, 3])
        c.put_data('node1', 0, 0, 'train', data)
        result = c.get_data('node1', 0, 0, 'train')
        assert np.array_equal(result, data)

    def test_get_missing(self):
        c = DataCache(maxsize=1024**3)
        assert c.get_data('no_exist', 0, 0, 'train') is None

    def test_clear_nodes(self):
        c = DataCache(maxsize=1024**3)
        c.put_data('a', 0, 0, 'train', np.array([1]))
        c.put_data('b', 0, 0, 'train', np.array([2]))
        c.clear_nodes(['a'])
        assert c.get_data('a', 0, 0, 'train') is None
        assert c.get_data('b', 0, 0, 'train') is not None

    def test_clear(self):
        c = DataCache(maxsize=1024**3)
        c.put_data('a', 0, 0, 'train', np.array([1]))
        c.clear()
        assert c.get_data('a', 0, 0, 'train') is None


class TestExperimenterInit:
    def test_path_created(self, exp):
        assert exp.path.exists()

    def test_data_wrapped(self, exp):
        assert exp.data is not None

    def test_splits_created(self, exp):
        assert exp.get_n_splits() == 2
        assert len(exp.outer_folds) == 2
        assert all(of.test_idx is not None for of in exp.outer_folds)

    def test_no_inner_split(self, exp):
        assert exp.get_n_splits_inner() == 1
        flow = _flow(exp)
        assert flow.data_source.valid_idx is None

    def test_with_inner_split(self, tmp_path, sample_data):
        e = Experimenter(
            data=sample_data,
            path=tmp_path / 'exp_inner',
            sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
            sp_v=KFold(n_splits=3, shuffle=True, random_state=42),
        )
        assert e.get_n_splits() == 2
        assert e.get_n_splits_inner() == 3
        flow = e.outer_folds[0].train_data_flows[0]
        assert flow.data_source.valid_idx is not None

    def test_create_path_exists(self, tmp_path, sample_data):
        path = tmp_path / 'existing'
        path.mkdir()
        with pytest.raises(RuntimeError):
            Experimenter.create(data=sample_data, path=path)

    def test_status_open(self, exp):
        assert exp.status == 'open'

    def test_pipeline_empty(self, exp):
        assert len(exp.pipeline.grps) == 0

    def test_data_key(self, tmp_path, sample_data):
        e = Experimenter(data=sample_data, path=tmp_path / 'dk', data_key='test_key')
        assert e.data_key == 'test_key'


class TestPipelineDelegation:
    def test_set_grp(self, exp):
        r = exp.set_grp('g1', role='stage')
        assert r['result'] == 'new'
        assert 'g1' in exp.pipeline.grps

    def test_set_node(self, exp):
        exp.set_grp('g1', role='stage', processor=StandardScaler,
                    method='transform', edges={'X': [(None, None)]})
        r = exp.set_node('n1', grp='g1')
        assert r['result'] == 'new'
        assert 'n1' in exp.pipeline.nodes

    def test_remove_node(self, exp):
        _setup_stage(exp)
        exp.remove_node('scaler')
        assert 'scaler' not in exp.pipeline.nodes

    def test_remove_grp(self, exp):
        exp.set_grp('g1', role='stage')
        exp.remove_grp('g1')
        assert 'g1' not in exp.pipeline.grps

    def test_rename_grp(self, exp):
        exp.set_grp('old', role='stage')
        exp.rename_grp('old', 'new')
        assert 'new' in exp.pipeline.grps
        assert 'old' not in exp.pipeline.grps

    def test_closed_status_blocks_modification(self, exp):
        exp.close()
        with pytest.raises(RuntimeError):
            exp.set_grp('g1', role='stage')


class TestBuild:
    def test_build_stage(self, exp):
        _setup_stage(exp)
        exp.build()
        flow = _flow(exp)
        assert 'scaler' in flow.node_objs
        assert flow.status('scaler') == 'built'

    def test_build_skips_built(self, exp):
        _setup_stage(exp)
        exp.build()
        flow = _flow(exp)
        build_id = flow.get_info('scaler')['build_id']
        exp.build()
        assert flow.get_info('scaler')['build_id'] == build_id

    def test_build_error_continues(self, exp):
        exp.set_grp('good', role='stage', processor=StandardScaler,
                    method='transform', edges={'X': [(None, ['f1'])]})
        exp.set_node('good_node', grp='good')
        exp.set_grp('bad', role='stage', processor=BadProcessor,
                    method='transform', edges={'X': [(None, ['f2'])]})
        exp.set_node('bad_node', grp='bad')
        exp.build()
        flow = _flow(exp)
        assert flow.status('good_node') == 'built'
        assert flow.status('bad_node') == 'error'

    def test_build_error_dict(self, exp):
        exp.set_grp('err', role='stage', processor=ErrorProcessor,
                    method='transform', edges={'X': [(None, ['f1'])]})
        exp.set_node('err_node', grp='err')
        exp.build()
        info = _flow(exp).get_info('err_node')
        err = info['error']
        assert err['type'] == 'TypeError'
        assert 'test error msg' in err['message']
        assert 'traceback' in err


class TestExp:
    def test_exp_head(self, exp):
        _setup_full(exp)
        exp.build()
        exp.exp()
        assert exp.get_status('dt') == 'built'

    def test_exp_skips_built(self, exp):
        _setup_full(exp)
        exp.build()
        exp.exp()
        store = _store(exp)
        build_id = store.get_info('dt')['build_id']
        exp.exp()
        assert store.get_info('dt')['build_id'] == build_id

    def test_exp_error(self, exp):
        exp.set_grp('bad_model', role='head', processor=BadPredictor,
                    method='predict', edges={'X': [(None, ['f1'])],
                                              'y': [(None, 'target')]})
        exp.set_node('bad_dt', grp='bad_model')
        exp.exp()
        assert exp.get_status('bad_dt') == 'error'

    def test_exp_with_collector(self, exp):
        _setup_full(exp)
        exp.build()
        mc = MetricCollector(
            'acc', Connector(),
            output_var=None,
            metric_func=accuracy_metric,
        )
        exp.add_collector(mc)
        exp.exp()
        assert mc.has('dt')


class TestCollectorManagement:
    def test_add_collector(self, exp):
        mc = MetricCollector('acc', Connector(),
                            output_var=None,
                            metric_func=dummy_metric)
        exp.add_collector(mc)
        assert exp.get_collector('acc') is not None
        assert mc.path is not None

    def test_add_collector_skip(self, exp):
        mc1 = MetricCollector('acc', Connector(),
                             output_var=None,
                             metric_func=dummy_metric)
        mc2 = MetricCollector('acc', Connector(),
                             output_var=None,
                             metric_func=dummy_metric)
        exp.add_collector(mc1)
        result = exp.add_collector(mc2, exist='skip')
        assert result is mc1

    def test_add_collector_error(self, exp):
        mc1 = MetricCollector('acc', Connector(),
                             output_var=None,
                             metric_func=dummy_metric)
        exp.add_collector(mc1)
        mc2 = MetricCollector('acc', Connector(),
                             output_var=None,
                             metric_func=dummy_metric)
        with pytest.raises(RuntimeError):
            exp.add_collector(mc2, exist='error')


class TestResetNodes:
    def test_reset_removes_node_dir(self, exp):
        _setup_stage(exp)
        exp.build()
        flow = _flow(exp)
        node_path = flow._node_path('scaler')
        assert node_path.exists()
        exp.reset_nodes(['scaler'])
        assert not node_path.exists()
        assert flow.status('scaler') is None

    def test_reset_clears_cache(self, exp):
        _setup_stage(exp)
        exp.build()
        exp.cache.put_data('scaler', 0, 0, 'train', np.array([1]))
        exp.reset_nodes(['scaler'])
        assert exp.cache.get_data('scaler', 0, 0, 'train') is None

    def test_set_node_replace_resets(self, exp):
        _setup_stage(exp)
        exp.build()
        flow = _flow(exp)
        assert flow.status('scaler') == 'built'
        exp.set_node('scaler', grp='scale', exist='replace')
        assert flow.status('scaler') is None


class TestRebuild:
    def test_build_with_rebuild_true(self, exp):
        _setup_stage(exp)
        exp.build()
        flow = _flow(exp)
        old_obj = flow.node_objs['scaler'][0]
        exp.build(rebuild=True)
        new_obj = _flow(exp).node_objs['scaler'][0]
        assert flow.status('scaler') == 'built'
        assert old_obj is not new_obj

    def test_set_node_replace_then_build(self, exp):
        _setup_stage(exp)
        exp.build()
        flow = _flow(exp)
        old_obj = flow.node_objs['scaler'][0]
        exp.set_node('scaler', grp='scale', exist='replace')
        exp.build()
        new_obj = _flow(exp).node_objs['scaler'][0]
        assert new_obj is not old_obj
        assert _flow(exp).status('scaler') == 'built'

    def test_exp_rebuilds_non_built_node(self, exp):
        _setup_full(exp)
        exp.build()
        exp.exp()
        assert exp.get_status('dt') == 'built'
        exp.reset_nodes(['dt'])
        assert exp.get_status('dt') is None
        exp.exp()
        assert exp.get_status('dt') == 'built'


class TestStateManagement:
    def test_open_close(self, exp):
        exp.close()
        assert exp.status == 'close'
        exp.open()
        assert exp.status == 'open'

    def test_finalize_head(self, exp):
        _setup_full(exp)
        exp.build()
        exp.exp()
        exp.finalize(['dt'])
        assert exp.get_status('dt') == 'finalized'

    def test_reinitialize(self, exp):
        _setup_full(exp)
        exp.build()
        exp.exp()
        exp.finalize(['dt'])
        exp.reinitialize(['dt'])
        assert exp.get_status('dt') is None

    def test_reopen_exp_status(self, exp):
        _setup_full(exp)
        exp.build()
        exp.exp()
        exp.close_exp()
        assert exp.status == 'closed'
        exp.reopen_exp()
        assert exp.status == 'open'

    def test_reopen_exp_collector_data_valid(self, exp):
        _setup_full(exp)
        exp.build()
        mc = MetricCollector('acc', Connector(edges = {'y': [(None, 'target')]}), output_var=None, metric_func=accuracy_metric)
        exp.add_collector(mc)
        exp.exp()
        assert mc.has('dt')
        first_result = mc.get_metrics_agg(None)[0]

        exp.close_exp()
        exp.reopen_exp()
        exp.exp()

        assert mc.has('dt')
        second_result = mc.get_metrics_agg(None)[0]
        assert second_result.shape == first_result.shape

    def test_reset_nodes_clears_collector_sub(self, exp):
        _setup_full(exp)
        exp.build()
        mc = MetricCollector('acc', Connector(), output_var=None, metric_func=accuracy_metric)
        exp.add_collector(mc)
        exp.exp()

        mc._buf['dt'] = [{'valid': 0.9}]
        exp.reset_nodes(['dt'])

        assert 'dt' not in mc._buf

    def test_close_exp_saves_status(self, exp, sample_data):
        _setup_full(exp)
        exp.build()
        exp.exp()
        exp.close_exp()

        loaded = Experimenter.load(exp.path, sample_data)
        assert loaded.status == 'closed'

    def test_reopen_exp_after_save_load(self, exp, sample_data):
        _setup_full(exp)
        exp.build()
        mc = MetricCollector('acc', Connector(edges = {'y': [(None, 'target')]}), output_var=None, metric_func=accuracy_metric)
        exp.add_collector(mc)
        exp.exp()
        first_result = mc.get_metrics_agg(None)[0]
        exp.close_exp()

        loaded = Experimenter.load(exp.path, sample_data)
        assert loaded.status == 'closed'
        loaded.reopen_exp()
        assert loaded.status == 'open'
        loaded.exp()

        mc2 = loaded.get_collector('acc')
        assert mc2.has('dt')
        second_result = mc2.get_metrics_agg(None)[0]
        assert second_result.shape == first_result.shape


class TestSaveLoad:
    def test_save_creates_file(self, exp):
        _setup_stage(exp)
        assert (exp.path / '__exp.pkl').exists()

    def test_load_restores(self, exp, sample_data):
        _setup_full(exp)
        exp.build()
        exp.exp()
        path = exp.path

        loaded = Experimenter.load(path, sample_data)
        assert set(loaded.pipeline.grps.keys()) == set(exp.pipeline.grps.keys())
        flow = loaded.outer_folds[0].train_data_flows[0]
        assert 'scaler' in flow.node_objs
        assert flow.status('scaler') == 'built'
        assert loaded.get_status('dt') == 'built'

    def test_load_data_key_mismatch(self, tmp_path, sample_data):
        e = Experimenter(data=sample_data, path=tmp_path / 'dk',
                        data_key='key_a')
        with pytest.raises(ValueError, match='data_key'):
            Experimenter.load(tmp_path / 'dk', sample_data, data_key='key_b')

    def test_load_preserves_splits(self, exp, sample_data):
        _setup_stage(exp)
        path = exp.path
        loaded = Experimenter.load(path, sample_data)
        assert loaded.get_n_splits() == exp.get_n_splits()
        assert loaded.get_n_splits_inner() == exp.get_n_splits_inner()


class TestPaths:
    def test_get_grp_path(self, exp):
        exp.set_grp('g1', role='stage')
        path = exp.get_grp_path('g1')
        assert path == exp.path / 'g1'

    def test_get_grp_path_nested(self, exp):
        exp.set_grp('parent', role='stage')
        exp.set_grp('child', role='stage', parent='parent')
        path = exp.get_grp_path('child')
        assert path == exp.path / 'parent' / 'child'

    def test_get_node_path(self, exp):
        _setup_stage(exp)
        path = exp.get_node_path('scaler')
        assert path == exp.path / 'scale' / 'scaler'


class TestGetStatus:
    def test_get_status_none_before_exp(self, exp):
        _setup_full(exp)
        exp.build()
        assert exp.get_status('dt') is None

    def test_get_status_built_after_exp(self, exp):
        _setup_full(exp)
        exp.build()
        exp.exp()
        assert exp.get_status('dt') == 'built'

    def test_get_status_finalized(self, exp):
        _setup_full(exp)
        exp.build()
        exp.exp()
        exp.finalize(['dt'])
        assert exp.get_status('dt') == 'finalized'

    def test_get_status_error(self, exp):
        exp.set_grp('bad_model', role='head', processor=BadPredictor,
                    method='predict', edges={'X': [(None, ['f1'])],
                                              'y': [(None, 'target')]})
        exp.set_node('bad_dt', grp='bad_model')
        exp.exp()
        assert exp.get_status('bad_dt') == 'error'


class TestNodeStore:
    def test_write_objs_and_status(self, tmp_path):
        store = NodeStore(tmp_path)
        node_path = tmp_path / 'node1'
        NodeStore.write_objs(node_path, object(), np.array([1, 2]), {'build_id': 'x'})
        assert store.status('node1') == 'built'

    def test_get_objs(self, tmp_path):
        store = NodeStore(tmp_path)
        node_path = tmp_path / 'node1'
        sc = StandardScaler()
        result = np.array([1.0, 2.0])
        NodeStore.write_objs(node_path, sc, result, {'build_id': 'abc'})
        got_obj, got_result, got_info = store.get_objs('node1')
        assert isinstance(got_obj, StandardScaler)
        assert np.array_equal(got_result, result)
        assert got_info['build_id'] == 'abc'
        assert got_info['status'] == 'built'

    def test_get_info(self, tmp_path):
        store = NodeStore(tmp_path)
        NodeStore.write_objs(tmp_path / 'node1', None, None, {'build_id': 'xyz'})
        info = store.get_info('node1')
        assert info['status'] == 'built'
        assert info['build_id'] == 'xyz'

    def test_get_obj_get_result(self, tmp_path):
        store = NodeStore(tmp_path)
        sc = StandardScaler()
        result = np.array([3.0])
        NodeStore.write_objs(tmp_path / 'node1', sc, result, {})
        assert isinstance(store.get_obj('node1'), StandardScaler)
        assert np.array_equal(store.get_result('node1'), result)

    def test_status_none_when_missing(self, tmp_path):
        store = NodeStore(tmp_path)
        assert store.status('missing') is None

    def test_write_info_error_status(self, tmp_path):
        store = NodeStore(tmp_path)
        node_path = tmp_path / 'node1'
        error_info = {
            'status': 'error',
            'build_id': 'e1',
            'error': {'type': 'ValueError', 'message': 'oops', 'traceback': '...'},
        }
        NodeStore.write_info(node_path, error_info)
        assert store.status('node1') == 'error'
        assert store.get_info('node1')['error']['type'] == 'ValueError'

    def test_finalize(self, tmp_path):
        store = NodeStore(tmp_path)
        node_path = tmp_path / 'node1'
        NodeStore.write_objs(node_path, None, None, {'build_id': 'y'})
        store.finalize('node1')
        assert store.status('node1') == 'finalized'
        assert not (node_path / 'obj.pkl').exists()
        assert not (node_path / 'result.pkl').exists()
        assert (node_path / 'info.pkl').exists()

    def test_reset_node(self, tmp_path):
        store = NodeStore(tmp_path)
        node_path = tmp_path / 'node1'
        NodeStore.write_objs(node_path, None, None, {})
        assert node_path.exists()
        store.reset_node('node1')
        assert not node_path.exists()
        assert store.status('node1') is None

    def test_info_cache_lazy(self, tmp_path):
        store = NodeStore(tmp_path)
        NodeStore.write_objs(tmp_path / 'node1', None, None, {'build_id': 'cached'})
        info1 = store.get_info('node1')
        info2 = store.get_info('node1')
        assert info1 is info2

    def test_reset_clears_cache(self, tmp_path):
        store = NodeStore(tmp_path)
        NodeStore.write_objs(tmp_path / 'node1', None, None, {})
        store.get_info('node1')  # populate cache
        store.reset_node('node1')
        assert store.status('node1') is None  # cache cleared, disk gone

    def test_dataflow_autoload(self, tmp_path):
        NodeStore.write_objs(tmp_path / 'node1', StandardScaler(), None, {'build_id': 'dl', 'edges': {'X': (None, ['X1', 'X2'])}})
        flow = DataFlow(tmp_path)
        assert 'node1' in flow.node_objs
        assert flow.status('node1') == 'built'

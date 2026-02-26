import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit, KFold

from mllabs._experimenter import Experimenter
from mllabs import Connector, MetricCollector, StackingCollector, ModelAttrCollector, OutputCollector


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
def built_exp(tmp_path, sample_data):
    e = Experimenter(
        data=sample_data,
        path=tmp_path / 'exp',
        sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
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
def built_exp_inner(tmp_path, sample_data):
    e = Experimenter(
        data=sample_data,
        path=tmp_path / 'exp_inner',
        sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        sp_v=KFold(n_splits=3, shuffle=True, random_state=42),
    )
    e.set_grp('model', role='head', processor=DecisionTreeClassifier,
              method='predict',
              edges={'X': [(None, ['f1', 'f2', 'f3'])], 'y': [(None, 'target')]},
              params={'max_depth': 3, 'random_state': 42})
    e.set_node('dt', grp='model')
    e.build()
    e.exp()
    return e


@pytest.fixture
def multi_head_exp(tmp_path, sample_data):
    e = Experimenter(
        data=sample_data,
        path=tmp_path / 'exp_multi',
        sp=ShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
    )
    e.set_grp('model', role='head', processor=DecisionTreeClassifier,
              method='predict',
              edges={'X': [(None, ['f1', 'f2', 'f3'])], 'y': [(None, 'target')]},
              params={'max_depth': 3, 'random_state': 42})
    e.set_node('dt1', grp='model')
    e.set_node('dt2', grp='model', params={'max_depth': 5})
    e.build()
    e.exp()
    return e


class TestConnector:
    def test_match_all(self):
        c = Connector()
        assert c.match('any_node', {}) is True

    def test_match_node_query_str(self):
        c = Connector(node_query='dt')
        assert c.match('dt1', {}) is True
        assert c.match('scaler', {}) is False

    def test_match_node_query_regex(self):
        c = Connector(node_query='^dt')
        assert c.match('dt1', {}) is True
        assert c.match('my_dt', {}) is False

    def test_match_node_query_list(self):
        c = Connector(node_query=['dt1', 'dt2'])
        assert c.match('dt1', {}) is True
        assert c.match('dt3', {}) is False

    def test_match_processor(self):
        c = Connector(processor=DecisionTreeClassifier)
        assert c.match('dt', {'processor': DecisionTreeClassifier}) is True
        assert c.match('dt', {'processor': StandardScaler}) is False

    def test_match_edges(self):
        edges_req = {'X': [(None, ['f1'])]}
        c = Connector(edges=edges_req)
        node_attrs = {'edges': {'X': [(None, ['f1']), ('s', None)], 'y': [(None, 'target')]}}
        assert c.match('dt', node_attrs) is True

    def test_match_edges_missing_key(self):
        c = Connector(edges={'z': [(None, None)]})
        assert c.match('dt', {'edges': {'X': [(None, None)]}}) is False

    def test_match_combined(self):
        c = Connector(node_query='dt', processor=DecisionTreeClassifier)
        assert c.match('dt1', {'processor': DecisionTreeClassifier}) is True
        assert c.match('dt1', {'processor': StandardScaler}) is False
        assert c.match('scaler', {'processor': DecisionTreeClassifier}) is False


class TestMetricCollector:
    def test_collect_basic(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.add_collector(mc)
        assert mc.has('dt')

    def test_get_metric(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.add_collector(mc)
        result = mc.get_metric('dt')
        assert isinstance(result, pd.Series)
        assert result.name == 'dt'
        assert len(result) > 0
        assert all(0 <= v <= 1 for v in result.values)

    def test_get_metrics(self, multi_head_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        multi_head_exp.add_collector(mc)
        result = mc.get_metrics()
        assert isinstance(result, pd.DataFrame)
        assert 'dt1' in result.index.get_level_values(0)
        assert 'dt2' in result.index.get_level_values(0)

    def test_get_metrics_with_node_filter(self, multi_head_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        multi_head_exp.add_collector(mc)
        result = mc.get_metrics(nodes=['dt1'])
        assert 'dt1' in result.index.get_level_values(0)
        assert 'dt2' not in result.index.get_level_values(0)

    def test_get_metrics_regex(self, multi_head_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        multi_head_exp.add_collector(mc)
        result = mc.get_metrics(nodes='dt1')
        assert len(result) > 0

    def test_get_metrics_agg(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.add_collector(mc)
        mean, std = mc.get_metrics_agg()
        assert isinstance(mean, pd.DataFrame)
        assert std is None

    def test_get_metrics_agg_with_std(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.add_collector(mc)
        mean, std = mc.get_metrics_agg(include_std=True)
        assert isinstance(mean, pd.DataFrame)
        assert isinstance(std, pd.DataFrame)

    def test_get_metrics_agg_inner_only(self, built_exp_inner):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp_inner.add_collector(mc)
        mean, std = mc.get_metrics_agg(inner_fold=True, outer_fold=False)
        assert isinstance(mean, pd.DataFrame)

    def test_get_metrics_agg_no_fold(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.add_collector(mc)
        result = mc.get_metrics_agg(inner_fold=False, outer_fold=False)
        assert isinstance(result, pd.DataFrame)

    def test_get_metrics_agg_invalid(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.add_collector(mc)
        with pytest.raises(ValueError):
            mc.get_metrics_agg(inner_fold=False, outer_fold=True)

    def test_include_train(self, built_exp):
        mc = MetricCollector('acc_train', Connector(), output_var=None,
                             metric_func=accuracy_metric, include_train=True)
        built_exp.add_collector(mc)
        result = mc.get_metric('dt')
        assert 'train_sub' in result.index.get_level_values(-1)

    def test_inner_split_metrics(self, built_exp_inner):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp_inner.add_collector(mc)
        result = mc.get_metric('dt')
        assert len(result) > 2

    def test_connector_filter(self, multi_head_exp):
        mc = MetricCollector('acc', Connector(node_query=['dt1']),
                             output_var=None, metric_func=accuracy_metric)
        multi_head_exp.add_collector(mc)
        assert mc.has('dt1')
        assert not mc.has('dt2')

    def test_reset_nodes(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.add_collector(mc)
        assert mc.has('dt')
        mc.reset_nodes(['dt'])
        assert not mc.has('dt')

    def test_save_load(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.add_collector(mc)
        loaded = MetricCollector.load(mc.path)
        assert loaded.has('dt')
        result_orig = mc.get_metric('dt')
        result_loaded = loaded.get_metric('dt')
        pd.testing.assert_series_equal(result_orig, result_loaded)

    def test_ad_hoc_collect(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.add_collector(mc)
        mc2 = MetricCollector('acc2', Connector(), output_var=None,
                              metric_func=dummy_metric)
        built_exp.add_collector(mc2)
        assert mc2.has('dt')
        result = mc2.get_metric('dt')
        assert all(v == 0.5 for v in result.values)


class TestStackingCollector:
    def test_collect_basic(self, built_exp):
        sc = StackingCollector('stk', Connector(), output_var=None,
                               experimenter=built_exp)
        built_exp.add_collector(sc)
        assert sc.has_node('dt')

    def test_get_dataset(self, built_exp):
        sc = StackingCollector('stk', Connector(
            edges={'y': [(None, 'target')]}
        ), output_var=None, experimenter=built_exp)
        built_exp.add_collector(sc)
        ds = sc.get_dataset()
        assert isinstance(ds, pd.DataFrame)
        assert len(ds) == int(len(built_exp.data.data) * 0.2) * 2 # ShuffleSplit, n_splits = 2, test_size = 0.2
        assert 'target' in ds.columns

    def test_get_dataset_no_target(self, built_exp):
        sc = StackingCollector('stk', Connector(), output_var=None,
                               experimenter=built_exp)
        built_exp.add_collector(sc)
        ds = sc.get_dataset(include_target=False)
        assert isinstance(ds, pd.DataFrame)
        assert 'target' not in ds.columns

    def test_get_dataset_multi_nodes(self, multi_head_exp):
        sc = StackingCollector('stk', Connector(
            edges={'y': [(None, 'target')]}
        ), output_var=None, experimenter=multi_head_exp)
        multi_head_exp.add_collector(sc)
        ds = sc.get_dataset()
        assert ds.shape[1] > 2

    def test_get_dataset_node_filter(self, multi_head_exp):
        sc = StackingCollector('stk', Connector(
            edges={'y': [(None, 'target')]}
        ), output_var=None, experimenter=multi_head_exp)
        multi_head_exp.add_collector(sc)
        ds = sc.get_dataset(nodes=['dt1'])
        assert isinstance(ds, pd.DataFrame)

    def test_method_mean(self, built_exp_inner):
        sc = StackingCollector('stk_mean', Connector(
            edges={'y': [(None, 'target')]}
        ), output_var=None, experimenter=built_exp_inner, method='mean')
        built_exp_inner.add_collector(sc)
        assert sc.has_node('dt')

    def test_reset_nodes(self, built_exp):
        sc = StackingCollector('stk', Connector(), output_var=None,
                               experimenter=built_exp)
        built_exp.add_collector(sc)
        assert sc.has_node('dt')
        sc.reset_nodes(['dt'])
        assert not sc.has_node('dt')

    def test_save_load(self, built_exp):
        sc = StackingCollector('stk', Connector(
            edges={'y': [(None, 'target')]}
        ), output_var=None, experimenter=built_exp)
        built_exp.add_collector(sc)
        loaded = StackingCollector.load(sc.path)
        assert loaded.has_node('dt')
        ds_orig = sc.get_dataset()
        ds_loaded = loaded.get_dataset()
        pd.testing.assert_frame_equal(ds_orig, ds_loaded)

    def test_index_preserved(self, built_exp):
        sc = StackingCollector('stk', Connector(
            edges={'y': [(None, 'target')]}
        ), output_var=None, experimenter=built_exp)
        built_exp.add_collector(sc)
        ds = sc.get_dataset()
        all_valid_idx = np.concatenate([
            built_exp.valid_idx_list[i]
            for i in range(built_exp.get_n_splits())
        ])
        expected_index = built_exp.data.data.index[all_valid_idx]
        pd.testing.assert_index_equal(ds.index, expected_index)


class TestModelAttrCollector:
    def test_collect_basic(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.add_collector(mac)
        assert mac.has('dt')

    def test_get_attr(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.add_collector(mac)
        result = mac.get_attr('dt')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_get_attr_idx(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.add_collector(mac)
        result = mac.get_attr('dt', idx=0)
        assert isinstance(result, list)

    def test_get_attrs(self, multi_head_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        multi_head_exp.add_collector(mac)
        result = mac.get_attrs()
        assert 'dt1' in result
        assert 'dt2' in result

    def test_get_attrs_agg(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.add_collector(mac)
        result = mac.get_attrs_agg('dt')
        assert isinstance(result, pd.Series)

    def test_get_attrs_agg_inner_only(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.add_collector(mac)
        result = mac.get_attrs_agg('dt', agg_inner=True, agg_outer=False)
        assert isinstance(result, pd.DataFrame)

    def test_get_attrs_agg_invalid(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.add_collector(mac)
        with pytest.raises(ValueError):
            mac.get_attrs_agg('dt', agg_inner=False, agg_outer=True)

    def test_not_mergeable(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('tree', Connector(processor=DecisionTreeClassifier),
                                result_key='tree',
                                adapter=DecisionTreeAdapter())
        built_exp.add_collector(mac)
        with pytest.raises(ValueError, match='not mergeable'):
            mac.get_attrs_agg('dt')

    def test_reset_nodes(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.add_collector(mac)
        mac.reset_nodes(['dt'])
        assert not mac.has('dt')

    def test_save_load(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.add_collector(mac)
        loaded = ModelAttrCollector.load(mac.path)
        assert loaded.has('dt')

    def test_auto_adapter(self):
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances')
        assert mac.adapter is not None

    def test_auto_adapter_invalid_key(self):
        with pytest.raises(RuntimeError):
            ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                               result_key='nonexistent_key')


class TestOutputCollector:
    def test_collect_basic(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.add_collector(oc)
        assert oc.has_node('dt')

    def test_get_output(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.add_collector(oc)
        result = oc.get_output('dt', 0, 0)
        assert 'output_valid' in result
        assert 'output_train' in result
        assert 'columns' in result

    def test_get_output_structure(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.add_collector(oc)
        result = oc.get_output('dt', 0, 0)
        assert isinstance(result['output_valid'], np.ndarray)
        assert isinstance(result['output_train'], tuple)
        assert len(result['output_train']) == 2

    def test_get_outputs(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.add_collector(oc)
        results = oc.get_outputs('dt')
        assert isinstance(results, dict)
        assert len(results) == built_exp.get_n_splits() * built_exp.get_n_splits_inner()
        for key in results:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_get_outputs_inner_split(self, built_exp_inner):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp_inner.add_collector(oc)
        results = oc.get_outputs('dt')
        n_expected = built_exp_inner.get_n_splits() * built_exp_inner.get_n_splits_inner()
        assert len(results) == n_expected

    def test_get_output_not_found(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.add_collector(oc)
        with pytest.raises(FileNotFoundError):
            oc.get_output('dt', 99, 99)

    def test_get_outputs_node_not_found(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.add_collector(oc)
        with pytest.raises(FileNotFoundError):
            oc.get_outputs('nonexistent')

    def test_reset_nodes(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.add_collector(oc)
        assert oc.has_node('dt')
        oc.reset_nodes(['dt'])
        assert not oc.has_node('dt')

    def test_save_load(self, built_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        built_exp.add_collector(oc)
        loaded = OutputCollector.load(oc.path)
        assert loaded.has_node('dt')
        result_orig = oc.get_output('dt', 0, 0)
        result_loaded = loaded.get_output('dt', 0, 0)
        np.testing.assert_array_equal(result_orig['output_valid'],
                                      result_loaded['output_valid'])

    def test_saved_nodes(self, multi_head_exp):
        oc = OutputCollector('out', Connector(), output_var=None)
        multi_head_exp.add_collector(oc)
        saved = oc._get_saved_nodes()
        assert 'dt1' in saved
        assert 'dt2' in saved


class TestCollectorWithExperimenter:
    def test_collect_skip_existing(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.add_collector(mc)
        metric_before = mc.get_metric('dt').copy()
        built_exp.collect(mc, exist='skip')
        metric_after = mc.get_metric('dt')
        pd.testing.assert_series_equal(metric_before, metric_after)

    def test_experimenter_save_load_with_collectors(self, built_exp, sample_data):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.add_collector(mc)
        path = built_exp.path

        loaded = Experimenter.load(path, sample_data)
        assert loaded.get_collector('acc') is not None
        loaded_mc = loaded.get_collector('acc')
        assert loaded_mc.has('dt')
        result_orig = mc.get_metric('dt')
        result_loaded = loaded_mc.get_metric('dt')
        pd.testing.assert_series_equal(result_orig, result_loaded)

    def test_reset_nodes_clears_collectors(self, built_exp):
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        built_exp.add_collector(mc)
        assert mc.has('dt')
        built_exp.reset_nodes(['dt'])
        assert not mc.has('dt')

    def test_multiple_collectors(self, built_exp):
        from mllabs.adapter import DecisionTreeAdapter
        mc = MetricCollector('acc', Connector(), output_var=None,
                             metric_func=accuracy_metric)
        oc = OutputCollector('out', Connector(), output_var=None)
        mac = ModelAttrCollector('fi', Connector(processor=DecisionTreeClassifier),
                                result_key='feature_importances',
                                adapter=DecisionTreeAdapter())
        built_exp.add_collector(mc)
        built_exp.add_collector(oc)
        built_exp.add_collector(mac)
        assert mc.has('dt')
        assert oc.has_node('dt')
        assert mac.has('dt')


class TestSHAPCollector:
    @pytest.fixture(autouse=True)
    def skip_if_no_shap(self):
        pytest.importorskip('shap')

    def _make_sc(self, exp):
        from mllabs import SHAPCollector
        sc = SHAPCollector('shap', Connector(processor=DecisionTreeClassifier))
        exp.add_collector(sc)
        return sc

    def test_collect_basic(self, built_exp):
        sc = self._make_sc(built_exp)
        assert sc.has_node('dt')

    def test_get_feature_importance_returns_list(self, built_exp):
        sc = self._make_sc(built_exp)
        result = sc.get_feature_importance('dt', 0)
        assert isinstance(result, list)
        assert len(result) == 1  # no inner split → 1 inner fold

    def test_get_feature_importance_series_structure(self, built_exp):
        sc = self._make_sc(built_exp)
        result = sc.get_feature_importance('dt', 0)
        s = result[0]
        assert isinstance(s, pd.Series)
        assert len(s.index) == 3
        assert (s >= 0).all()

    def test_get_feature_importance_inner_order(self, built_exp_inner):
        sc = self._make_sc(built_exp_inner)
        result = sc.get_feature_importance('dt', 0)
        assert len(result) == 3  # KFold n_splits=3
        assert [s.name for s in result] == [0, 1, 2]

    def test_get_feature_importance_agg_default_returns_series(self, built_exp):
        sc = self._make_sc(built_exp)
        result = sc.get_feature_importance_agg('dt')
        assert isinstance(result, pd.Series)
        assert len(result.index) == 3
        assert (result >= 0).all()

    def test_get_feature_importance_agg_outer_none_returns_dataframe(self, built_exp):
        sc = self._make_sc(built_exp)
        result = sc.get_feature_importance_agg('dt', agg_outer=None)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 2)  # 3 features x 2 outer folds

    def test_get_feature_importance_agg_inner_none_multiindex(self, built_exp_inner):
        sc = self._make_sc(built_exp_inner)
        result = sc.get_feature_importance_agg('dt', agg_inner=None)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(result.columns, pd.MultiIndex)
        assert result.shape == (3, 6)  # 3 features x (2 outer * 3 inner)

    def test_get_feature_importance_agg_callable(self, built_exp):
        sc = self._make_sc(built_exp)
        result = sc.get_feature_importance_agg('dt', agg_inner=np.mean, agg_outer=np.mean)
        assert isinstance(result, pd.Series)

    def test_reset_nodes(self, built_exp):
        sc = self._make_sc(built_exp)
        assert sc.has_node('dt')
        sc.reset_nodes(['dt'])
        assert not sc.has_node('dt')

    def test_save_load(self, built_exp):
        sc = self._make_sc(built_exp)
        from mllabs import SHAPCollector
        loaded = SHAPCollector.load(sc.path)
        assert loaded.has_node('dt')
        orig = sc.get_feature_importance_agg('dt')
        loaded_result = loaded.get_feature_importance_agg('dt')
        pd.testing.assert_series_equal(orig, loaded_result)


class TestBaseCollector:
    def test_get_nodes_none(self):
        from mllabs.collector._base import Collector
        c = Collector('test', Connector())
        result = c._get_nodes(None, ['a', 'b', 'c'])
        assert result == ['a', 'b', 'c']

    def test_get_nodes_list(self):
        from mllabs.collector._base import Collector
        c = Collector('test', Connector())
        result = c._get_nodes(['a', 'c'], ['a', 'b', 'c'])
        assert result == ['a', 'c']

    def test_get_nodes_list_filter(self):
        from mllabs.collector._base import Collector
        c = Collector('test', Connector())
        result = c._get_nodes(['a', 'x'], ['a', 'b', 'c'])
        assert result == ['a']

    def test_get_nodes_regex(self):
        from mllabs.collector._base import Collector
        c = Collector('test', Connector())
        result = c._get_nodes('dt', ['dt1', 'dt2', 'scaler'])
        assert result == ['dt1', 'dt2']

    def test_get_nodes_invalid_type(self):
        from mllabs.collector._base import Collector
        c = Collector('test', Connector())
        with pytest.raises(ValueError):
            c._get_nodes(123, ['a', 'b'])

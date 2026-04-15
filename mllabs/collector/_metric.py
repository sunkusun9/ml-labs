import pickle
import re

import pandas as pd

from ._base import Collector
from .._node_processor import resolve_columns


class MetricCollector(Collector):
    _SAVE_EXCLUDE = {'_buf': dict, '_cache': dict}

    def __init__(self, name, connector, output_var, metric_func, include_train=False):
        super().__init__(name, connector)
        self.output_var = output_var
        self.metric_func = metric_func
        self.include_train = include_train
        self._cache = {}  # {node: {(outer_idx, inner_idx): result}}

    def collect(self, context):
        cols = resolve_columns(context['output_test'], self.output_var)
        if len(cols) == 0:
            return None

        prd_test = context['output_test'].select_columns(cols)
        result = {'test': self.metric_func(context['input'][2]['y'].data, prd_test.data)}

        if self.include_train and context.get('output_train') is not None:
            result['train'] = self.metric_func(
                context['input'][0]['y'].data, context['output_train'].select_columns(cols).data
            )
            if context['output_valid'] is not None:
                result['valid'] = self.metric_func(
                    context['input'][1]['y'].data, context['output_valid'].select_columns(cols).data
                )

        return result

    def _flush_outer(self, node, outer_idx, inner_list):
        node_cache = self._cache.setdefault(node, {})
        for inner_idx, r in enumerate(inner_list):
            node_cache[(outer_idx, inner_idx)] = r
        if self.path is not None and self._n_outer is not None and len(node_cache) == self._n_outer * self._n_inner:
            self.path.mkdir(parents=True, exist_ok=True)
            with open(self.path / f'{node}.pkl', 'wb') as f:
                pickle.dump(node_cache, f)

    def _load_results(self, node):
        if node in self._cache:
            return self._cache[node]
        if self.path is None:
            return None
        p = self.path / f'{node}.pkl'
        if not p.exists():
            return None
        with open(p, 'rb') as f:
            result = pickle.load(f)
        self._cache[node] = result
        return result

    def has_node(self, node):
        if node in self._cache:
            return True
        if self.path is None:
            return False
        return (self.path / f'{node}.pkl').exists()

    def has(self, node):
        return self.has_node(node)

    def reset_nodes(self, nodes):
        node_set = set(nodes)
        self._buf = {k: v for k, v in self._buf.items() if k not in node_set}
        for node in nodes:
            self._cache.pop(node, None)
            if self.path is not None:
                p = self.path / f'{node}.pkl'
                if p.exists():
                    p.unlink()

    def _get_nodes(self, nodes, available):
        if nodes is None:
            return available
        if isinstance(nodes, list):
            return [n for n in nodes if n in set(available)]
        return [n for n in available if re.search(nodes, n)]

    def _get_saved_nodes(self):
        if self.path is None:
            return list(self._cache.keys())
        return [f.stem for f in self.path.glob('*.pkl') if f.name != '__config.pkl']

    def get_metric(self, node):
        data = self._load_results(node)
        if data is None:
            return None
        outer_idxs = sorted(set(k[0] for k in data.keys()))
        l = []
        for idx in outer_idxs:
            sub = [data[(idx, inner_idx)] for inner_idx in sorted(
                k[1] for k in data if k[0] == idx
            )]
            l.append(
                pd.concat([pd.Series(j, name=str(no)) for no, j in enumerate(sub)], axis=1).unstack()
            )
        return pd.concat(l, axis=1).unstack(level=[0, 1]).rename(node)

    def get_metrics(self, nodes=None):
        node_names = self._get_nodes(nodes, self._get_saved_nodes())
        results = [self.get_metric(node) for node in node_names]
        results = [r for r in results if r is not None]
        if not results:
            return None
        return pd.concat(results, axis=1).T

    def get_metrics_agg(self, nodes=None, inner_fold=True, outer_fold=True, include_std=False):
        if outer_fold and not inner_fold:
            raise ValueError("")
        df = self.get_metrics(nodes)
        if inner_fold:
            df_agg_mean = df.stack(level=1, future_stack=True).groupby(level=0).mean()
            if include_std:
                df_agg_std = df.stack(level=1, future_stack=True).groupby(level=0).std()
            else:
                df_agg_std = None
            if outer_fold:
                if include_std:
                    df_agg_std = df_agg_mean.stack(level=0, future_stack=True).groupby(level=0).std()
                df_agg_mean = df_agg_mean.stack(level=0, future_stack=True).groupby(level=0).mean()
            return df_agg_mean, df_agg_std
        return df

    def get_properties(self):
        return {
            'need_output_train': self.include_train,
            'need_output_test': True,
            'need_process_data': False,
        }
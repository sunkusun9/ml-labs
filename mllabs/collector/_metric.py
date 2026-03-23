import pickle
import pandas as pd

from ._base import Collector
from .._node_processor import resolve_columns


class MetricCollector(Collector):
    def __init__(self, name, connector, output_var, metric_func, include_train=False):
        super().__init__(name, connector)
        self.output_var = output_var
        self.metric_func = metric_func
        self.include_train = include_train

    def collect(self, context):
        if context['input'] is None or 'y' not in context['input']:
            return None

        (true_t, true_tv), true_v = context['input']['y']
        cols = resolve_columns(context['output_valid'], self.output_var)
        if len(cols) == 0:
            return None

        prd_v = context['output_valid'].select_columns(cols)
        result = {'valid': self.metric_func(true_v.data, prd_v.data)}

        if self.include_train and context.get('output_train') is not None:
            prd_t = context['output_train'][0].select_columns(cols)
            result['train_sub'] = self.metric_func(true_t.data, prd_t.data)
            if true_tv is not None:
                prd_tv = context['output_train'][1].select_columns(cols)
                result['valid_sub'] = self.metric_func(true_tv.data, prd_tv.data)

        return result

    def reset_nodes(self, nodes):
        super().reset_nodes(nodes)

    def save(self):
        if self.path is None:
            return
        self._ensure_path()
        data = {
            'name': self.name,
            'connector': self.connector,
            'output_var': self.output_var,
            'metric_func': self.metric_func,
            'include_train': self.include_train,
            '_node_paths': self._node_paths,
        }
        with open(self.path / '__config.pkl', 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path / '__config.pkl', 'rb') as f:
            data = pickle.load(f)
        obj = cls(
            name=data['name'],
            connector=data['connector'],
            output_var=data['output_var'],
            metric_func=data['metric_func'],
            include_train=data['include_train'],
        )
        obj._node_paths = data.get('_node_paths', {})
        obj.path = path
        return obj

    def get_metric(self, node):
        data = self._load_collect_results(node)
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
        return pd.concat([self.get_metric(node) for node in node_names], axis=1).T

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

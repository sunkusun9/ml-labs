import pickle
import pandas as pd

from ._base import Collector
from ..adapter import get_adapter


class ModelAttrCollector(Collector):
    def __init__(self, name, connector, result_key, adapter=None, params=None):
        super().__init__(name, connector)
        self.result_key = result_key
        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = get_adapter(connector.processor)
            if self.result_key not in self.adapter.result_objs:
                raise RuntimeError("")
        self.params = params or {}

    def collect(self, context):
        result_func = self.adapter.result_objs[self.result_key][0]
        return result_func(context['processor'], **self.params)

    def reset_nodes(self, nodes):
        super().reset_nodes(nodes)

    def save(self):
        if self.path is None:
            return
        self._ensure_path()
        data = {
            'name': self.name,
            'connector': self.connector,
            'result_key': self.result_key,
            'adapter': self.adapter,
            'params': self.params,
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
            result_key=data['result_key'],
            adapter=data.get('adapter'),
            params=data['params'],
        )
        obj._node_paths = data.get('_node_paths', {})
        obj.path = path
        return obj

    def _is_mergeable(self):
        if self.adapter is None or self.result_key not in self.adapter.result_objs:
            return False
        return self.adapter.result_objs[self.result_key][1]

    def get_attr(self, node, idx=None):
        data = self._load_collect_results(node)
        outer_idxs = sorted(set(k[0] for k in data.keys()))
        result = []
        for oi in outer_idxs:
            inner_list = [data[(oi, ii)] for ii in sorted(k[1] for k in data if k[0] == oi)]
            result.append(inner_list)
        if idx is not None:
            return result[idx]
        return result

    def get_attrs(self, nodes=None):
        node_names = self._get_nodes(nodes, self._get_saved_nodes())
        return {node: self.get_attr(node) for node in node_names}

    def get_attrs_agg(self, node, agg_inner=True, agg_outer=True):
        if agg_outer and not agg_inner:
            raise ValueError("agg_outer requires agg_inner to be True")
        if not self._is_mergeable():
            raise ValueError(f"Result '{self.result_key}' is not mergeable across folds")
        results = self.get_attr(node)
        l = []
        for no, inner_list in enumerate(results):
            l.append(
                pd.concat([j.rename(no_i) for no_i, j in enumerate(inner_list)], axis=1).stack().rename(no)
            )
        df = pd.concat(l, axis=1)
        if agg_inner:
            df = df.groupby(level=[i for i in range(len(df.index.levels) - 1)]).mean()
            if agg_outer:
                return df.mean(axis=1)
        return df

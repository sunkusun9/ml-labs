import pickle
import pandas as pd

from ._base import Collector
from ..adapter import get_adapter


class ModelAttrCollector(Collector):
    """Collects model attributes (e.g. feature importances) for each fold.

    Args:
        name (str): Collector name.
        connector (Connector): Node matching criteria. Used to infer the
            adapter from ``connector.processor`` when *adapter* is ``None``.
        result_key (str): Key in the adapter's ``result_objs`` dict (e.g.
            ``'feature_importances'``).
        adapter (ModelAdapter, optional): Explicit adapter. Auto-inferred
            from ``connector.processor`` if omitted.
        params (dict, optional): Extra keyword arguments forwarded to the
            result extractor function.
    """

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
        self.results = {}
        self._sub = {}

    def _start(self, node):
        self.results[node] = []
        self._sub[node] = []

    def _collect(self, node, idx, inner_idx, context):
        result_func = self.adapter.result_objs[self.result_key][0]
        result = result_func(context['processor'], **self.params)
        self._sub[node].append(result)

    def _end_idx(self, node, idx):
        self.results[node].append(self._sub[node])
        self._sub[node] = []

    def _end(self, node):
        if len(self.results[node]) == 0:
            del self.results[node]
        if node in self._sub:
            del self._sub[node]
        self.save()

    def has_node(self, node):
        return node in self.results

    def reset_nodes(self, nodes):
        for node in nodes:
            if node in self.results:
                del self.results[node]
            if node in self._sub:
                del self._sub[node]
        self.save()

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
            'results': self.results,
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
        obj.results = data['results']
        obj.path = path
        return obj

    def _is_mergeable(self):
        if self.adapter is None or self.result_key not in self.adapter.result_objs:
            return False
        return self.adapter.result_objs[self.result_key][1]

    def get_attr(self, node, idx=None):
        if idx is not None:
            return self.results[node][idx]
        return self.results[node]

    def get_attrs(self, nodes=None):
        node_names = self._get_nodes(nodes, self.results.keys())
        return {node: self.results[node] for node in node_names}

    def get_attrs_agg(self, node, agg_inner=True, agg_outer=True):
        """Return aggregated model attributes across folds.

        Only valid for mergeable result types (``result_objs[key][1] == True``).

        Args:
            node (str): Node name.
            agg_inner (bool): Average inner folds. Required when ``agg_outer=True``.
            agg_outer (bool): Average outer folds after inner aggregation.
                Returns a ``pd.Series`` when both are ``True``.

        Returns:
            pd.Series | pd.DataFrame: Aggregated result.

        Raises:
            ValueError: If the result type is not mergeable or
                ``agg_outer=True`` while ``agg_inner=False``.
        """
        if agg_outer and not agg_inner:
            raise ValueError("agg_outer requires agg_inner to be True")
        if not self._is_mergeable():
            raise ValueError(f"Result '{self.result_key}' is not mergeable across folds")
        l = list()
        for no, inner_list in enumerate(self.results[node]):
            l.append(
                pd.concat([j.rename(no_i) for no_i, j in enumerate(inner_list)], axis=1).stack().rename(no)
            )
        df = pd.concat(l, axis=1)
        if agg_inner:
            df = df.groupby(level=[i for i in range(len(df.index.levels) - 1)]).mean()
            if agg_outer:
                return df.mean(axis=1)
        return df

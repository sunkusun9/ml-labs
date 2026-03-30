import os
import pickle as pkl
from abc import ABC, abstractmethod
from pathlib import Path

from ._node_processor import resolve_columns
# from ._expobj import _build_sub


class DataSourceProvider(ABC):
    @abstractmethod
    def get_train(self):
        """Returns train_data as DataWrapper."""

    @abstractmethod
    def get_valid(self):
        """Returns valid_data (train-time monitoring, e.g. early stopping) as DataWrapper or None."""

class DataFlow:
    """Single-fold data transformation through stage nodes.

    Loads one processor per stage node from disk at ``path``.
    Transforms source data through the stage graph given edges.
    No build functionality.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.node_objs = {}    # {name: (obj, result, info)}
        self._node_edges = {}  # {name: edges dict}
        self.load()

    def get_objs_file(self, node_name):
        return self.path / node_name / 'obj.pkl'

    def set_objs(self, node_name, obj, result, info):
        self.node_objs[node_name] = (obj, result, info)
        if info.get('edges') is not None:
            self._node_edges[node_name] = info['edges']

    def load_objs(self, node_name):
        with open(self.get_objs_file(node_name), 'rb') as f:
            obj, result, info = pkl.load(f)
        self.set_objs(node_name, obj, result, info)
        return obj, result, info

    def load(self):
        if not self.path.is_dir():
            return
        for node_dir in sorted(self.path.iterdir()):
            if not node_dir.is_dir():
                continue
            if self.get_objs_file(node_dir.name).exists():
                self.load_objs(node_dir.name)

    def get_data(self, source_data, edges):
        """Transform source_data through stage nodes per edges.

        Args:
            source_data: DataWrapper — raw input at DataSource level
            edges: {key: [(node_name, var), ...]}

        Returns:
            {key: data} flat dict
        """
        result = {}
        for key, edge_list in edges.items():
            parts = []
            for node_name, var in edge_list:
                data = self._resolve(source_data, node_name)
                if data is None:
                    continue
                if var is not None:
                    obj = self.node_objs[node_name][0] if node_name in self.node_objs else None
                    cols = resolve_columns(data, var, processor=obj)
                    data = data.select_columns(cols)
                parts.append(data)
            if parts:
                result[key] = type(parts[0]).concat(parts, axis=1) if len(parts) > 1 else parts[0]
        return result

    def _resolve(self, source_data, node_name):
        if node_name is None:
            return source_data
        if node_name not in self.node_objs or node_name not in self._node_edges:
            return None
        obj, result, info = self.node_objs[node_name]
        edges = self._node_edges[node_name]
        key = 'X' if 'X' in edges else next(iter(edges))
        parts = []
        for src_node, var in edges[key]:
            data = self._resolve(source_data, src_node)
            if data is not None and var is not None:
                src_obj = self.node_objs.get(src_node, (None,))[0] if src_node in self.node_objs else None
                cols = resolve_columns(data, var, processor=src_obj)
                data = data.select_columns(cols)
            if data is not None:
                parts.append(data)
        if not parts:
            return None
        T = type(parts[0])
        return obj.process(T.concat(parts, axis=1) if len(parts) > 1 else parts[0])


class TrainDataFlow(DataFlow):
    """Single (outer, inner) fold data flow with stage build capability.

    Args:
        path: Per-fold storage directory
        data_source: DataSourceProvider providing train/valid/test raw data
        cache: DataCache shared instance (optional)
        cache_key: Key for cache lookups, e.g. (outer_idx, inner_idx)
    """

    def __init__(self, path, data_source, cache=None, cache_key=None):
        self.data_source = data_source
        self.cache = cache
        self.cache_key = cache_key
        super().__init__(path)

    @staticmethod
    def write_objs(file, obj_data):
        os.makedirs(file.parent, exist_ok=True)
        with open(file, 'wb') as f:
            pkl.dump(obj_data, f)

    def get_available_stages(self, pipeline):
        """Returns stage node names that this DataFlow can produce output for."""
        return [
            n for n in pipeline._get_affected_nodes([None])
            if n is not None
            and n in self.node_objs
            and pipeline.grps[pipeline.nodes[n].grp].role == 'stage'
        ]

    def get_missing_stages(self, pipeline):
        """Returns stage node names that are in the pipeline but not yet built in this DataFlow."""
        return [
            n for n in pipeline._get_affected_nodes([None])
            if n is not None
            and n not in self.node_objs
            and pipeline.grps[pipeline.nodes[n].grp].role == 'stage'
        ]

    def get_train(self, edges):
        """{key: data} train output resolved via edges."""
        return self._get_resolved_edges(edges, self._resolve_train)

    def get_valid(self, edges):
        """{key: data} valid (train-time monitoring) output resolved via edges."""
        return self._get_resolved_edges(
            edges, lambda n: self._resolve_via_process(n, 'valid', self.data_source.get_valid)
        )

    def _get_resolved_edges(self, edges, resolve_fn):
        result = {}
        for key, edge_list in edges.items():
            parts = []
            for node_name, var in edge_list:
                data = resolve_fn(node_name)
                if var is not None:
                    obj = self.node_objs.get(node_name, (None,))[0] if node_name in self.node_objs else None
                    cols = resolve_columns(data, var, processor=obj)
                    data = data.select_columns(cols)
                parts.append(data)
            if parts:
                result[key] = type(parts[0]).concat(parts, axis=1) if len(parts) > 1 else parts[0]
        return result

    def _resolve_train(self, node_name):
        """Returns train data for node_name (fit_transform result)."""
        if node_name is None:
            return self.data_source.get_train()
        if node_name not in self.node_objs:
            raise RuntimeError(f"Stage '{node_name}' not built in this flow")

        if self.cache is not None and self.cache_key is not None:
            cached = self.cache.get_data(node_name, 'train', self.cache_key)
            if cached is not None:
                return cached

        obj, result, info = self.node_objs[node_name]
        edges = self._node_edges[node_name]
        key = 'X' if 'X' in edges else next(iter(edges))

        trains = []
        for src_node, var in edges[key]:
            t = self._resolve_train(src_node)
            if var is not None:
                src_obj = self.node_objs.get(src_node, (None,))[0] if src_node in self.node_objs else None
                cols = resolve_columns(t, var, processor=src_obj)
                t = t.select_columns(cols)
            trains.append(t)

        T = type(trains[0])
        train_in = T.concat(trains, axis=1) if len(trains) > 1 else trains[0]
        train_out = result if result is not None else obj.process(train_in)

        if self.cache is not None and self.cache_key is not None:
            self.cache.put_data(node_name, 'train', self.cache_key, train_out)

        return train_out

    def _resolve_via_process(self, node_name, typ, source_fn):
        """Returns data for node_name via cache check → process. Used for valid and test."""
        if node_name is None:
            return source_fn()
        if node_name not in self.node_objs:
            raise RuntimeError(f"Stage '{node_name}' not built in this flow")

        if self.cache is not None and self.cache_key is not None:
            cached = self.cache.get_data(node_name, typ, self.cache_key)
            if cached is not None:
                return cached

        obj, result, info = self.node_objs[node_name]
        edges = self._node_edges[node_name]
        key = 'X' if 'X' in edges else next(iter(edges))

        parts = []
        for src_node, var in edges[key]:
            d = self._resolve_via_process(src_node, typ, source_fn)
            if d is not None and var is not None:
                src_obj = self.node_objs.get(src_node, (None,))[0] if src_node in self.node_objs else None
                cols = resolve_columns(d, var, processor=src_obj)
                d = d.select_columns(cols)
            if d is not None:
                parts.append(d)

        if not parts:
            return None

        T = type(parts[0])
        data_in = T.concat(parts, axis=1) if len(parts) > 1 else parts[0]
        data_out = obj.process(data_in)

        if self.cache is not None and self.cache_key is not None:
            self.cache.put_data(node_name, typ, self.cache_key, data_out)

        return data_out

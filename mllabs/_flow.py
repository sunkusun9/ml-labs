import os
import pickle as pkl
from abc import ABC, abstractmethod
from pathlib import Path

from ._node_processor import resolve_columns
from ._expobj import _build_sub


class DataSourceProvider(ABC):
    @abstractmethod
    def get_train(self):
        """Returns (train_data, train_v_data) as DataWrapper pair."""

    @abstractmethod
    def get_valid(self):
        """Returns valid_data as DataWrapper."""


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
        self._load()

    def _load(self):
        if not self.path.is_dir():
            return
        for node_dir in sorted(self.path.iterdir()):
            if not node_dir.is_dir():
                continue
            obj_file = node_dir / 'obj.pkl'
            attrs_file = node_dir / 'attrs.pkl'
            if obj_file.exists():
                with open(obj_file, 'rb') as f:
                    self.node_objs[node_dir.name] = pkl.load(f)
            if attrs_file.exists():
                with open(attrs_file, 'rb') as f:
                    self._node_edges[node_dir.name] = pkl.load(f)

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
        data_source: DataSourceProvider providing train/valid raw data
        cache: DataCache shared instance (optional)
        cache_key: Key for cache lookups, e.g. (outer_idx, inner_idx)
    """

    def __init__(self, path, data_source, cache=None, cache_key=None):
        self.data_source = data_source
        self.cache = cache
        self.cache_key = cache_key
        super().__init__(path)

    def build_stage(self, node_attrs, logger):
        """Build a stage node for this fold and persist to disk.

        Args:
            node_attrs: dict with name, processor, method, adapter, params, edges
            logger: Logger instance
        """
        name = node_attrs['name']
        edges = node_attrs['edges']
        fit_process = node_attrs['method'] in ['fit_transform', 'fit_predict']

        data_dict = self._make_data_dict(edges)
        obj, result, info = _build_sub(node_attrs, data_dict, fit_process, logger)

        node_path = self.path / name
        os.makedirs(node_path, exist_ok=True)
        with open(node_path / 'obj.pkl', 'wb') as f:
            pkl.dump((obj, result, info), f)
        with open(node_path / 'attrs.pkl', 'wb') as f:
            pkl.dump(edges, f)

        self.node_objs[name] = (obj, result, info)
        self._node_edges[name] = edges

    def _make_data_dict(self, edges):
        """{key: ((train, train_v), valid)} for _build_sub."""
        data_dict = {}
        for key, edge_list in edges.items():
            trains, train_vs, valids = [], [], []
            for node_name, var in edge_list:
                train, train_v = self._resolve_train(node_name)
                valid = self._resolve_valid(node_name)
                if var is not None:
                    src_obj = self.node_objs.get(node_name, (None,))[0] if node_name in self.node_objs else None
                    cols = resolve_columns(train, var, processor=src_obj)
                    train = train.select_columns(cols)
                    if train_v is not None:
                        train_v = train_v.select_columns(cols)
                    valid = valid.select_columns(cols)
                trains.append(train)
                if train_v is not None:
                    train_vs.append(train_v)
                valids.append(valid)
            if trains:
                T = type(trains[0])
                data_dict[key] = (
                    (T.concat(trains, axis=1) if len(trains) > 1 else trains[0],
                     T.concat(train_vs, axis=1) if train_vs else None),
                    T.concat(valids, axis=1) if len(valids) > 1 else valids[0],
                )
        return data_dict

    def get_train(self, edges):
        """{key: data} train output resolved via edges."""
        result = {}
        for key, edge_list in edges.items():
            parts = []
            for node_name, var in edge_list:
                train, _ = self._resolve_train(node_name)
                if var is not None:
                    obj = self.node_objs.get(node_name, (None,))[0] if node_name in self.node_objs else None
                    cols = resolve_columns(train, var, processor=obj)
                    train = train.select_columns(cols)
                parts.append(train)
            if parts:
                result[key] = type(parts[0]).concat(parts, axis=1) if len(parts) > 1 else parts[0]
        return result

    def get_valid(self, edges):
        """{key: data} valid output resolved via edges."""
        result = {}
        for key, edge_list in edges.items():
            parts = []
            for node_name, var in edge_list:
                valid = self._resolve_valid(node_name)
                if var is not None:
                    obj = self.node_objs.get(node_name, (None,))[0] if node_name in self.node_objs else None
                    cols = resolve_columns(valid, var, processor=obj)
                    valid = valid.select_columns(cols)
                parts.append(valid)
            if parts:
                result[key] = type(parts[0]).concat(parts, axis=1) if len(parts) > 1 else parts[0]
        return result

    def _resolve_train(self, node_name):
        """Returns (train, train_v) for node_name."""
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

        trains, train_vs = [], []
        for src_node, var in edges[key]:
            t, tv = self._resolve_train(src_node)
            if var is not None:
                src_obj = self.node_objs.get(src_node, (None,))[0] if src_node in self.node_objs else None
                cols = resolve_columns(t, var, processor=src_obj)
                t = t.select_columns(cols)
                if tv is not None:
                    tv = tv.select_columns(cols)
            trains.append(t)
            if tv is not None:
                train_vs.append(tv)

        T = type(trains[0])
        train_in = T.concat(trains, axis=1) if len(trains) > 1 else trains[0]
        train_v_in = T.concat(train_vs, axis=1) if train_vs else None

        train_out = result if result is not None else obj.process(train_in)
        train_v_out = obj.process(train_v_in) if train_v_in is not None else None

        if self.cache is not None and self.cache_key is not None:
            self.cache.put_data(node_name, 'train', self.cache_key, (train_out, train_v_out))

        return train_out, train_v_out

    def _resolve_valid(self, node_name):
        """Returns valid data for node_name."""
        if node_name is None:
            return self.data_source.get_valid()
        if node_name not in self.node_objs:
            raise RuntimeError(f"Stage '{node_name}' not built in this flow")

        if self.cache is not None and self.cache_key is not None:
            cached = self.cache.get_data(node_name, 'valid', self.cache_key)
            if cached is not None:
                return cached

        obj, result, info = self.node_objs[node_name]
        edges = self._node_edges[node_name]
        key = 'X' if 'X' in edges else next(iter(edges))

        valids = []
        for src_node, var in edges[key]:
            v = self._resolve_valid(src_node)
            if var is not None:
                src_obj = self.node_objs.get(src_node, (None,))[0] if src_node in self.node_objs else None
                cols = resolve_columns(v, var, processor=src_obj)
                v = v.select_columns(cols)
            valids.append(v)

        T = type(valids[0])
        valid_in = T.concat(valids, axis=1) if len(valids) > 1 else valids[0]
        valid_out = obj.process(valid_in)

        if self.cache is not None and self.cache_key is not None:
            self.cache.put_data(node_name, 'valid', self.cache_key, valid_out)

        return valid_out

import pickle
import numpy as np

from ._base import Collector
from .._node_processor import resolve_columns
from .._data_wrapper import DataWrapper


class StackingCollector(Collector):
    def __init__(self, name, connector, output_var, experimenter, method='mean'):
        super().__init__(name, connector)
        self.output_var = output_var
        self.method = method

        self._data_cls = type(experimenter.data)
        self._index = self._build_index(experimenter)
        self._target, self._target_columns = self._build_target(experimenter)

    def _build_index(self, experimenter):
        all_valid_idx = np.concatenate([
            experimenter.valid_idx_list[i]
            for i in range(experimenter.get_n_splits())
        ])
        return experimenter.data.iloc(all_valid_idx).get_index()

    def _build_target(self, experimenter):
        target_vars = self.connector.edges.get('y') if self.connector.edges else None
        if target_vars is None:
            return None, None

        target_list = []
        target_columns = None
        temp_edges = {'_target': target_vars}
        for idx in range(experimenter.get_n_splits()):
            iterator = experimenter.get_data_valid(idx, temp_edges)
            aggregated = DataWrapper.simple(
                data_dict['_target'] for data_dict in iterator
            )
            if target_columns is None:
                target_columns = aggregated.get_columns()
            target_list.append(aggregated.to_array())
        if type(target_columns) is str:
            target_columns = [target_columns]
        return np.concatenate(target_list, axis=0), target_columns

    def collect(self, context):
        cols = resolve_columns(context['output_valid'], self.output_var)
        if len(cols) == 0:
            return None
        return context['output_valid'].select_columns(cols)

    def _aggregate(self, iterator):
        if self.method == 'simple':
            return self._data_cls.simple(iterator)
        elif self.method == 'mean':
            return self._data_cls.mean(iterator)
        elif self.method == 'mode':
            return self._data_cls.mode(iterator)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def reset_nodes(self, nodes):
        super().reset_nodes(nodes)

    def save(self):
        if self.path is None:
            return
        self._ensure_path()
        config = {
            'name': self.name,
            'connector': self.connector,
            'output_var': self.output_var,
            'method': self.method,
            '_data_cls': self._data_cls,
            '_index': self._index,
            '_target': self._target,
            '_target_columns': self._target_columns,
            '_node_paths': self._node_paths,
        }
        with open(self.path / '__config.pkl', 'wb') as f:
            pickle.dump(config, f)

    @classmethod
    def load(cls, path):
        with open(path / '__config.pkl', 'rb') as f:
            config = pickle.load(f)
        obj = object.__new__(cls)
        obj.name = config['name']
        obj.connector = config['connector']
        obj.output_var = config['output_var']
        obj.method = config['method']
        obj._data_cls = config['_data_cls']
        obj._index = config['_index']
        obj._target = config['_target']
        obj._target_columns = config['_target_columns']
        obj._node_paths = config.get('_node_paths', {})
        obj.path = path
        obj.warnings = []
        return obj

    def _load_node_data(self, node):
        """Load and aggregate inner folds per outer fold, then concat outer folds."""
        p = self._node_paths[node]
        outer_files = sorted(p.glob('_collect_*.pkl'), key=lambda x: int(x.stem.rsplit('_', 1)[1]))
        arrays, columns = [], None
        for f in outer_files:
            with open(f, 'rb') as fp:
                inner_results = pickle.load(fp)
            valid_results = [r for r in inner_results if r is not None]
            if not valid_results:
                continue
            aggregated = self._aggregate(iter(valid_results))
            if columns is None:
                columns = aggregated.get_columns()
                if type(columns) is str:
                    columns = [columns]
            arrays.append(aggregated.to_array())
        if not arrays:
            return None, columns
        return np.concatenate(arrays, axis=0), columns

    def get_dataset(self, nodes=None, include_target=True):
        node_names = self._get_nodes(nodes, self._get_saved_nodes())

        node_data = []
        column_names = []
        for n in node_names:
            data, columns = self._load_node_data(n)
            if data is None:
                continue
            node_data.append(data)
            column_names.extend(columns)

        all_data = np.concatenate(node_data, axis=1)
        all_columns = list(column_names)

        wrapped_data = self._data_cls.from_output(all_data, all_columns, self._index)
        if include_target and self._target is not None:
            wrapped_data = self._data_cls.concat([
                wrapped_data,
                self._data_cls.from_output(self._target, self._target_columns, self._index)
            ], axis=1)

        return wrapped_data.to_native()

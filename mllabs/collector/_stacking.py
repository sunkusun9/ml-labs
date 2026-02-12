import pickle
import numpy as np

from ._base import Collector
from .._node_processor import resolve_columns
from .._data_wrapper import DataWrapper


class StackingCollector(Collector):
    def __init__(self, name, connector, output_var, method='mean', include_target=True):
        super().__init__(name, connector)
        self.output_var = output_var
        self.method = method
        self.include_target = include_target
        self.columns = {}
        self._buffer = {}
        self._mem_data = {}

    def _start(self, node):
        self._buffer[node] = []

    def _collect(self, node, idx, inner_idx, context):
        cols = resolve_columns(context['output_valid'], self.output_var)
        if len(cols) == 0:
            return
        valid_output = context['output_valid'].select_columns(cols)
        self._buffer[node].append(valid_output)
        if inner_idx == 0:
            self.columns[node] = valid_output.get_columns()

    def _end_idx(self, node, idx):
        if len(self._buffer[node]) == 0:
            return
        aggregated = self._aggregate(iter(self._buffer[node]))
        arr = aggregated.to_array()

        if self.path is not None:
            self._ensure_path()
            file_path = self.path / f"{node}.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    existing = pickle.load(f)
                existing['data'] = np.concatenate([existing['data'], arr])
            else:
                existing = {'data': arr, 'columns': self.columns[node]}
            with open(file_path, 'wb') as f:
                pickle.dump(existing, f)
        else:
            if node not in self._mem_data:
                self._mem_data[node] = {'data': arr, 'columns': self.columns[node]}
            else:
                self._mem_data[node]['data'] = np.concatenate([self._mem_data[node]['data'], arr])

        self._buffer[node] = []

    def _end(self, node):
        if node in self._buffer:
            del self._buffer[node]

    def _aggregate(self, iterator):
        if self.method == 'simple':
            return DataWrapper.simple(iterator)
        elif self.method == 'mean':
            return DataWrapper.mean(iterator)
        elif self.method == 'mode':
            return DataWrapper.mode(iterator)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def has_node(self, node):
        if node in self._mem_data:
            return True
        if self.path is not None:
            return (self.path / f"{node}.pkl").exists()
        return False

    def reset_nodes(self, nodes):
        for node in nodes:
            if self.path is not None:
                file_path = self.path / f"{node}.pkl"
                if file_path.exists():
                    file_path.unlink()
            if node in self._mem_data:
                del self._mem_data[node]
            if node in self.columns:
                del self.columns[node]

    def save(self):
        if self.path is None:
            return
        self._ensure_path()
        config = {
            'name': self.name,
            'connector': self.connector,
            'output_var': self.output_var,
            'method': self.method,
            'include_target': self.include_target,
        }
        with open(self.path / '__config.pkl', 'wb') as f:
            pickle.dump(config, f)

    @classmethod
    def load(cls, path):
        with open(path / '__config.pkl', 'rb') as f:
            config = pickle.load(f)
        obj = cls(
            name=config['name'],
            connector=config['connector'],
            output_var=config['output_var'],
            method=config['method'],
            include_target=config['include_target'],
        )
        obj.path = path
        return obj

    def _load_node_data(self, node):
        if self.path is not None:
            file_path = self.path / f"{node}.pkl"
            if not file_path.exists():
                raise FileNotFoundError(f"Stacking data not found: {file_path}")
            with open(file_path, 'rb') as f:
                node_info = pickle.load(f)
            return node_info['data'], node_info['columns']
        else:
            if node not in self._mem_data:
                raise KeyError(f"Stacking data not found in memory: {node}")
            return self._mem_data[node]['data'], self._mem_data[node]['columns']

    def _get_saved_nodes(self):
        if self.path is not None:
            if not self.path.exists():
                return []
            return [f.stem for f in self.path.glob("*.pkl") if not f.stem.startswith('__')]
        else:
            return list(self._mem_data.keys())

    def _build_sort_order(self, experimenter):
        all_valid_idx = np.concatenate([
            experimenter.valid_idx_list[i]
            for i in range(experimenter.get_n_splits())
        ])
        return np.argsort(all_valid_idx)

    def _build_target_value(self, experimenter):
        target_vars = self.connector.edges.get('y') if self.connector.edges else None
        if target_vars is None:
            raise ValueError("Connector must define 'y' edges for target")

        target_list = []
        temp_edges = {'_target': target_vars}
        for idx in range(experimenter.get_n_splits()):
            iterator = experimenter.get_data_valid(idx, temp_edges)
            def extract_target(it):
                for data_dict in it:
                    yield data_dict['_target']
            aggregated = DataWrapper.simple(extract_target(iterator))
            target_list.append(aggregated)

        wrapper_cls = type(target_list[0])
        return wrapper_cls.concat(target_list, axis=0)

    def get_dataset(self, experimenter, nodes=None):
        node_names = self._get_nodes(nodes, self._get_saved_nodes())
        sort_order = self._build_sort_order(experimenter)

        node_data = []
        column_names = []
        for i in node_names:
            data, columns = self._load_node_data(i)
            node_data.append(data)
            column_names.extend(columns)

        if self.include_target:
            target = self._build_target_value(experimenter)
            wrapper_cls = type(target)
            return wrapper_cls.concat([
                wrapper_cls.from_output(
                    np.concatenate(node_data, axis=1),
                    column_names, target.get_index()
                )
            ] + [target], axis=1).iloc(sort_order).data
        else:
            wrapper_cls = type(experimenter.data)
            all_valid_idx = np.concatenate([
                experimenter.valid_idx_list[i]
                for i in range(experimenter.get_n_splits())
            ])
            index = experimenter.data.iloc(all_valid_idx).get_index()
            return wrapper_cls.from_output(
                np.concatenate(node_data, axis=1),
                column_names, index
            ).iloc(sort_order).data

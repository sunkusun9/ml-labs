import pickle
import numpy as np

from ._base import Collector
from .._node_processor import resolve_columns


class ProcessCollector(Collector):
    def __init__(self, name, connector, ext_data, experimenter, output_var=None, method='mean'):
        super().__init__(name, connector)
        self.output_var = output_var
        self.method = method
        self._ext_data = ext_data
        self._experimenter = experimenter
        self._data_cls = type(experimenter.data)
        self._input_cache = {}  # {(node, idx): [inner0_input, ...]} transient per outer fold

    def collect(self, context):
        node = context['node_attrs']['name']
        idx = context['idx']
        inner_idx = context['inner_idx']

        cache_key = (node, idx)
        if cache_key not in self._input_cache:
            self._input_cache[cache_key] = list(
                self._experimenter.process_ext(self._ext_data, node, idx)
            )
        inputs = self._input_cache[cache_key]

        if inner_idx >= len(inputs) or inputs[inner_idx] is None:
            return None

        output = context['processor'].process(inputs[inner_idx])
        if self.output_var is not None:
            cols = resolve_columns(output, self.output_var, processor=context['processor'])
            output = output.select_columns(cols)
        return output

    def reset_nodes(self, nodes):
        for node in nodes:
            for key in list(self._input_cache.keys()):
                if key[0] == node:
                    del self._input_cache[key]
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
        obj._node_paths = config.get('_node_paths', {})
        obj._input_cache = {}
        obj._ext_data = None
        obj._experimenter = None
        obj.path = path
        obj.warnings = []
        return obj

    def _aggregate(self, iterator):
        if self.method == 'mean':
            return self._data_cls.mean(iterator)
        elif self.method == 'mode':
            return self._data_cls.mode(iterator)
        elif self.method == 'simple':
            return self._data_cls.simple(iterator)
        raise ValueError(f"Unsupported method: {self.method}")

    def _load_node_data(self, node, agg):
        """Load per-outer-fold data: aggregate inner folds, return list of outer fold arrays."""
        p = self._node_paths[node]
        outer_files = sorted(p.glob('_collect_*.pkl'), key=lambda x: int(x.stem.rsplit('_', 1)[1]))
        arrays, cols = [], None
        for f in outer_files:
            with open(f, 'rb') as fp:
                inner_results = pickle.load(fp)
            valid_results = [r for r in inner_results if r is not None]
            if not valid_results:
                continue
            if cols is None:
                cols = valid_results[0].get_columns()
            if len(valid_results) == 1:
                aggregated = valid_results[0]
            else:
                aggregated = self._aggregate(iter(valid_results))
            arrays.append(aggregated.to_array())

        wrapped = [self._data_cls.from_output(arr, cols) for arr in arrays]
        if len(wrapped) == 1:
            return wrapped[0], cols
        if agg == 'mean':
            return self._data_cls.mean(iter(wrapped)), cols
        elif agg == 'mode':
            return self._data_cls.mode(iter(wrapped)), cols
        elif agg == 'simple':
            return self._data_cls.simple(iter(wrapped)), cols
        raise ValueError(f"Unsupported agg: {agg}")

    def get_output(self, nodes=None, agg='mean'):
        node_names = self._get_nodes(nodes, self._get_saved_nodes())

        node_data, column_names = [], []
        for node in node_names:
            result, cols = self._load_node_data(node, agg)
            node_data.append(result.to_array())
            column_names.extend(cols if cols is not None else [])

        all_data = np.concatenate(node_data, axis=1)
        return self._data_cls.from_output(all_data, column_names).to_native()

import pickle
import shutil

import numpy as np

from ._base import Collector
from .._node_processor import resolve_columns


class ProcessCollector(Collector):
    """Collects predictions on external (test) data for each matched node.

    For each matched head node, passes the external data through upstream
    stage processors (via :meth:`~mllabs.Experimenter.process_ext`) and
    calls the fitted processor to produce predictions. Inner-fold predictions
    are aggregated per outer fold; outer-fold predictions are aggregated on
    query.

    Args:
        name (str): Collector name.
        connector (Connector): Node matching criteria.
        ext_data: External dataset (pandas/polars/numpy) to predict on.
        experimenter (Experimenter): Used to run upstream stage transforms
            via ``process_ext``.
        output_var: Column selector applied to processor output.
        method (str): Inner-fold aggregation — ``'mean'`` (default),
            ``'mode'``, or ``'simple'``.
    """

    def __init__(self, name, connector, ext_data, experimenter, output_var=None, method='mean'):
        super().__init__(name, connector)
        self.output_var = output_var
        self.method = method
        self._ext_data = ext_data
        self._experimenter = experimenter
        self._data_cls = type(experimenter.data)
        self.columns = {}
        self._buffer = {}
        self._input_cache = {}
        self._mem_data = {}

    def _start(self, node):
        self._buffer[node] = []
        self._input_cache[node] = {}

    def _collect(self, node, idx, inner_idx, context):
        if idx not in self._input_cache[node]:
            self._input_cache[node][idx] = list(
                self._experimenter.process_ext(self._ext_data, node, idx)
            )
        inputs = self._input_cache[node][idx]
        if inner_idx >= len(inputs) or inputs[inner_idx] is None:
            return

        output = context['processor'].process(inputs[inner_idx])
        if self.output_var is not None:
            cols = resolve_columns(output, self.output_var, processor=context['processor'])
            output = output.select_columns(cols)
        if inner_idx == 0:
            self.columns[node] = output.get_columns()
        self._buffer[node].append(output)

    def _end_idx(self, node, idx):
        self._input_cache[node].pop(idx, None)
        if not self._buffer[node]:
            return
        aggregated = self._aggregate(iter(self._buffer[node]))
        arr = aggregated.to_array()

        if self.path is not None:
            self._ensure_path()
            node_dir = self.path / node
            node_dir.mkdir(parents=True, exist_ok=True)
            with open(node_dir / f"{idx}.pkl", 'wb') as f:
                pickle.dump({'data': arr, 'columns': self.columns[node]}, f)
        else:
            if node not in self._mem_data:
                self._mem_data[node] = []
            self._mem_data[node].append(arr)

        self._buffer[node] = []

    def _end(self, node):
        self._buffer.pop(node, None)
        self._input_cache.pop(node, None)

    def _aggregate(self, iterator):
        if self.method == 'mean':
            return self._data_cls.mean(iterator)
        elif self.method == 'mode':
            return self._data_cls.mode(iterator)
        elif self.method == 'simple':
            return self._data_cls.simple(iterator)
        raise ValueError(f"Unsupported method: {self.method}")

    def has_node(self, node):
        if node in self._mem_data:
            return True
        if self.path is not None:
            node_dir = self.path / node
            return node_dir.exists() and any(node_dir.glob("*.pkl"))
        return False

    def reset_nodes(self, nodes):
        for node in nodes:
            self._mem_data.pop(node, None)
            self.columns.pop(node, None)
            self._buffer.pop(node, None)
            self._input_cache.pop(node, None)
            if self.path is not None:
                node_dir = self.path / node
                if node_dir.exists():
                    shutil.rmtree(node_dir)

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
            'columns': self.columns,
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
        obj.columns = config['columns']
        obj._buffer = {}
        obj._input_cache = {}
        obj._mem_data = {}
        obj._ext_data = None
        obj._experimenter = None
        obj.path = path
        obj.warnings = []
        return obj

    def _get_saved_nodes(self):
        if self.path is not None:
            if not self.path.exists():
                return []
            return [d.name for d in self.path.iterdir() if d.is_dir()]
        return list(self._mem_data.keys())

    def _load_node_data(self, node, agg):
        if self.path is not None:
            node_dir = self.path / node
            if not node_dir.exists():
                raise FileNotFoundError(f"Process data not found: {node_dir}")
            arrays, cols = [], None
            for f in sorted(node_dir.glob("*.pkl"), key=lambda p: int(p.stem)):
                with open(f, 'rb') as fp:
                    entry = pickle.load(fp)
                arrays.append(entry['data'])
                cols = entry['columns']
        else:
            if node not in self._mem_data:
                raise KeyError(f"Process data not found in memory: {node}")
            arrays = self._mem_data[node]
            cols = self.columns.get(node)

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
        """Return aggregated test predictions, optionally for multiple nodes.

        Args:
            nodes: Node query — ``None`` (all), list, or regex str.
            agg (str): Outer-fold aggregation — ``'mean'`` (default),
                ``'mode'``, or ``'simple'``.

        Returns:
            Native DataFrame with columns from all matched nodes concatenated.
        """
        node_names = self._get_nodes(nodes, self._get_saved_nodes())

        node_data, column_names = [], []
        for node in node_names:
            result, cols = self._load_node_data(node, agg)
            node_data.append(result.to_array())
            column_names.extend(cols if cols is not None else [])

        all_data = np.concatenate(node_data, axis=1)
        return self._data_cls.from_output(all_data, column_names).to_native()

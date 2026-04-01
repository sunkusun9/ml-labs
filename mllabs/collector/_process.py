import pickle
import re
import shutil

import numpy as np

from ._base import Collector
from .._node_processor import resolve_columns


class ProcessCollector(Collector):
    _SAVE_EXCLUDE = {
        '_buf': dict,
        '_input_cache': dict,
        '_ext_data': lambda: None,
        '_experimenter': lambda: None,
    }

    def __init__(self, name, connector, ext_data, experimenter, output_var=None, method='mean'):
        super().__init__(name, connector)
        self.output_var = output_var
        self.method = method
        self._data_cls = type(experimenter.data)
        self._ext_data = ext_data
        self._experimenter = experimenter
        self._input_cache = {}  # {(node, outer_idx): [inner0_input, ...]} transient

    def collect(self, context):
        node = context['node_attrs']['name']
        idx = context['outer_idx']
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

    def _aggregate(self, iterator):
        if self.method == 'mean':
            return self._data_cls.mean(iterator)
        elif self.method == 'mode':
            return self._data_cls.mode(iterator)
        elif self.method == 'simple':
            return self._data_cls.simple(iterator)
        raise ValueError(f"Unsupported method: {self.method}")

    def _flush_outer(self, node, outer_idx, inner_list):
        self._input_cache.pop((node, outer_idx), None)
        valid_results = [r for r in inner_list if r is not None]
        if not valid_results:
            return
        aggregated = valid_results[0] if len(valid_results) == 1 else self._aggregate(iter(valid_results))
        node_dir = self.path / node
        node_dir.mkdir(parents=True, exist_ok=True)
        with open(node_dir / f'{outer_idx}.pkl', 'wb') as f:
            pickle.dump({'data': aggregated.to_array(), 'columns': aggregated.get_columns()}, f)

    def has_node(self, node):
        if self.path is None:
            return False
        p = self.path / node
        return p.is_dir() and any(p.glob('*.pkl'))

    def has(self, node):
        return self.has_node(node)

    def reset_nodes(self, nodes):
        node_set = set(nodes)
        self._buf = {k: v for k, v in self._buf.items() if k not in node_set}
        self._input_cache = {k: v for k, v in self._input_cache.items() if k[0] not in node_set}
        for node in nodes:
            p = self.path / node
            if p.exists():
                shutil.rmtree(p)

    def _get_saved_nodes(self):
        if self.path is None:
            return []
        return [p.name for p in self.path.iterdir()
                if p.is_dir() and any(p.glob('*.pkl'))]

    def _get_nodes(self, nodes, available):
        if nodes is None:
            return available
        if isinstance(nodes, list):
            return [n for n in nodes if n in set(available)]
        return [n for n in available if re.search(nodes, n)]

    def _load_node(self, node, agg):
        p = self.path / node
        outer_files = sorted(p.glob('*.pkl'), key=lambda x: int(x.stem))
        arrays, cols = [], None
        for f in outer_files:
            with open(f, 'rb') as fp:
                saved = pickle.load(fp)
            if cols is None:
                cols = saved['columns']
            arrays.append(saved['data'])
        if len(arrays) == 1:
            return self._data_cls.from_output(arrays[0], cols), cols
        wrapped = [self._data_cls.from_output(a, cols) for a in arrays]
        if agg == 'mean':
            return self._data_cls.mean(iter(wrapped)), cols
        elif agg == 'mode':
            return self._data_cls.mode(iter(wrapped)), cols
        elif agg == 'simple':
            return self._data_cls.simple(iter(wrapped)), cols
        raise ValueError(f"Unsupported agg: {agg}")

    def get_output(self, nodes=None, agg='mean'):
        node_names = self._get_nodes(nodes, self._get_saved_nodes())
        arrays, columns = [], []
        for node in node_names:
            result, cols = self._load_node(node, agg)
            arrays.append(result.to_array())
            columns.extend(cols if cols is not None else [])
        all_data = np.concatenate(arrays, axis=1)
        return self._data_cls.from_output(all_data, columns).to_native()

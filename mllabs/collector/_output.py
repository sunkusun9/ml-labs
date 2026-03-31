import pickle
import re
import shutil

from ._base import Collector
from .._node_processor import resolve_columns


class OutputCollector(Collector):
    _SAVE_EXCLUDE = {'_buf': dict}

    def __init__(self, name, connector, output_var, include_target=True):
        super().__init__(name, connector)
        self.output_var = output_var
        self.include_target = include_target

    def collect(self, context):
        cols = resolve_columns(context['output_valid'], self.output_var)
        if len(cols) == 0:
            return None

        output_valid = context['output_valid'].select_columns(cols)
        train_sub = context['output_train'][0].select_columns(cols)
        valid_sub = context['output_train'][1]
        if valid_sub is not None:
            valid_sub = valid_sub.select_columns(cols)

        return {
            'output_train': (
                train_sub.to_array(),
                valid_sub.to_array() if valid_sub is not None else None
            ),
            'output_valid': output_valid.to_array(),
            'columns': output_valid.get_columns(),
        }

    def push(self, node, outer_idx, inner_idx, result):
        node_dir = self.path / node
        node_dir.mkdir(parents=True, exist_ok=True)
        with open(node_dir / f'{outer_idx}_{inner_idx}.pkl', 'wb') as f:
            pickle.dump(result, f)

    def has_node(self, node):
        if self.path is None:
            return False
        p = self.path / node
        return p.is_dir() and any(p.glob('*.pkl'))

    def has(self, node):
        return self.has_node(node)

    def reset_nodes(self, nodes):
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

    def get_output(self, node, idx, inner_idx):
        with open(self.path / node / f'{idx}_{inner_idx}.pkl', 'rb') as f:
            return pickle.load(f)

    def get_outputs(self, node):
        p = self.path / node
        result = {}
        for f in p.glob('*.pkl'):
            outer_idx, inner_idx = map(int, f.stem.split('_'))
            with open(f, 'rb') as fp:
                result[(outer_idx, inner_idx)] = pickle.load(fp)
        return result

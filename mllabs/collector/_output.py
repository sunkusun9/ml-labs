import pickle

from ._base import Collector
from .._node_processor import resolve_columns


class OutputCollector(Collector):
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
            'include_target': self.include_target,
            '_node_paths': self._node_paths,
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
            include_target=config['include_target'],
        )
        obj._node_paths = config.get('_node_paths', {})
        obj.path = path
        return obj

    def get_output(self, node, idx, inner_idx):
        p = self._node_paths[node]
        collect_file = p / f'_collect_{idx}.pkl'
        with open(collect_file, 'rb') as f:
            inner_results = pickle.load(f)
        return inner_results[inner_idx]

    def get_outputs(self, node):
        data = self._load_collect_results(node)
        return {k: v for k, v in data.items() if v is not None}

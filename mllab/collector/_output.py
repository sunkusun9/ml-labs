import pickle
import shutil

from ._base import Collector
from .._node_processor import resolve_columns


class OutputCollector(Collector):
    def __init__(self, name, connector, output_var, include_target=True):
        super().__init__(name, connector)
        self.output_var = output_var
        self.include_target = include_target

    def _collect(self, node, idx, inner_idx, context):
        cols = resolve_columns(context['output_valid'], self.output_var)
        if len(cols) == 0:
            return

        output_valid = context['output_valid'].select_columns(cols)
        train_sub = context['output_train'][0].select_columns(cols)
        valid_sub = context['output_train'][1]
        if valid_sub is not None:
            valid_sub = valid_sub.select_columns(cols)

        entry = {
            'output_train': (
                train_sub.to_array(),
                valid_sub.to_array() if valid_sub is not None else None
            ),
            'output_valid': output_valid.to_array(),
            'columns': output_valid.get_columns(),
        }

        self._ensure_path()
        node_dir = self.path / node
        node_dir.mkdir(parents=True, exist_ok=True)
        with open(node_dir / f"{idx}_{inner_idx}.pkl", 'wb') as f:
            pickle.dump(entry, f)

    def has_node(self, node):
        if self.path is not None:
            node_dir = self.path / node
            return node_dir.exists() and any(node_dir.glob("*.pkl"))
        return False

    def reset_nodes(self, nodes):
        for node in nodes:
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
            include_target=config['include_target'],
        )
        obj.path = path
        return obj

    def _get_saved_nodes(self):
        if self.path is None or not self.path.exists():
            return []
        return [d.name for d in self.path.iterdir() if d.is_dir()]

    def get_output(self, node, idx, inner_idx):
        file_path = self.path / node / f"{idx}_{inner_idx}.pkl"
        if not file_path.exists():
            raise FileNotFoundError(f"Output data not found: {file_path}")
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def get_outputs(self, node):
        node_dir = self.path / node
        if not node_dir.exists():
            raise FileNotFoundError(f"Node directory not found: {node_dir}")
        results = {}
        for f in sorted(node_dir.glob("*.pkl")):
            parts = f.stem.split('_')
            idx, inner_idx = int(parts[0]), int(parts[1])
            with open(f, 'rb') as fp:
                results[(idx, inner_idx)] = pickle.load(fp)
        return results

import re
import shutil
import pickle


class Collector:
    def __init__(self, name, connector):
        self.name = name
        self.connector = connector
        self.path = None
        self.warnings = []
        self._node_paths = {}  # {node_name: collector_data_dir (node_path / self.name)}

    def collect(self, context):
        return None

    def has(self, node):
        return self.has_node(node)

    def has_node(self, node):
        if node not in self._node_paths:
            return False
        p = self._node_paths[node]
        return p.exists() and any(p.glob('_collect_*.pkl'))

    def reset_nodes(self, nodes):
        for node in nodes:
            if node in self._node_paths:
                p = self._node_paths[node]
                if p.exists():
                    shutil.rmtree(p)
                del self._node_paths[node]
        self.save()

    def _load_collect_results(self, node):
        """Load all outer fold results. Returns {(idx, inner_idx): result}."""
        p = self._node_paths[node]
        results = {}
        for f in sorted(p.glob('_collect_*.pkl'), key=lambda x: int(x.stem.rsplit('_', 1)[1])):
            idx = int(f.stem.rsplit('_', 1)[1])
            with open(f, 'rb') as fp:
                inner_results = pickle.load(fp)
            for inner_idx, result in enumerate(inner_results):
                results[(idx, inner_idx)] = result
        return results

    def _get_saved_nodes(self):
        return [node for node, p in self._node_paths.items()
                if p.exists() and any(p.glob('_collect_*.pkl'))]

    def _ensure_path(self):
        if self.path is not None and not self.path.exists():
            self.path.mkdir(parents=True, exist_ok=True)

    def save(self):
        pass

    @classmethod
    def load(cls, path):
        raise NotImplementedError

    def _get_nodes(self, nodes, available_nodes):
        if nodes is None:
            return list(available_nodes)
        elif isinstance(nodes, list):
            return [n for n in nodes if n in available_nodes]
        elif isinstance(nodes, str):
            pat = re.compile(nodes)
            return [k for k in available_nodes if k is not None and pat.search(k)]
        else:
            raise ValueError(f"nodes must be None, list, or str, got {type(nodes)}")

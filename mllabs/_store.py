import os
import shutil
import pickle as pkl
from pathlib import Path


class NodeStore:
    """On-disk artifact manager for one (outer, inner) fold.

    Node artifacts are stored at {path}/{node_name}/obj.pkl.
    No in-memory caching.

    Args:
        path: (outer, inner) fold path.
    """

    def __init__(self, path):
        self.path = Path(path)

    def _node_path(self, name):
        return self.path / name

    def get_objs_file(self, name):
        return self._node_path(name) / 'obj.pkl'

    def status(self, name):
        node_path = self._node_path(name)
        if not os.path.isdir(node_path):
            return None
        if (node_path / 'finalized.pkl').exists():
            return 'finalized'
        if (node_path / 'obj.pkl').exists():
            return 'built'
        return None

    @classmethod
    def write_obj(cls, node_path, obj, result, info, finalize=False):
        os.makedirs(node_path, exist_ok=True)
        if finalize:
            with open(node_path / 'finalized.pkl', 'wb') as f:
                pkl.dump(info, f)
        else:
            with open(node_path / 'obj.pkl', 'wb') as f:
                pkl.dump((obj, result, info), f)

    def save_obj(self, name, obj, result, info, finalize=False):
        self.write_obj(self._node_path(name), obj, result, info, finalize)

    def finalize(self, name):
        node_path = self._node_path(name)
        obj_file = node_path / 'obj.pkl'
        with open(obj_file, 'rb') as f:
            _, _, spec = pkl.load(f)
        with open(node_path / 'finalized.pkl', 'wb') as f:
            pkl.dump(spec, f)
        obj_file.unlink()

    def reset_node(self, name):
        node_path = self._node_path(name)
        if os.path.isdir(node_path):
            shutil.rmtree(node_path)

    def get_obj(self, name):
        with open(self._node_path(name) / 'obj.pkl', 'rb') as f:
            return pkl.load(f)

import json
import os
import pickle as pkl


class ArtifactStore:
    """On-demand artifact manager for Head nodes in one (outer, inner) fold.

    Uses the same path structure as TrainDataFlow: {outer_path}/{inner_idx}/.
    Node artifacts are stored at {path}/{node_name}/obj.pkl.
    No in-memory caching.

    Args:
        path: (outer, inner) fold path — same as TrainDataFlow path.
    """

    def __init__(self, path):
        self.path = path

    def _node_path(self, name):
        return self.path / name

    def status(self, name):
        node_path = self._node_path(name)
        if not os.path.isdir(node_path):
            return None
        if (node_path / 'error.txt').exists():
            return 'error'
        if (node_path / 'finalized.pkl').exists():
            return 'finalized'
        if (node_path / 'obj.pkl').exists():
            return 'built'
        return None

    def get_error(self, name):
        error_path = self._node_path(name) / 'error.txt'
        if error_path.exists():
            with open(error_path) as f:
                return json.load(f)
        return None

    def set_error(self, name, error_info):
        node_path = self._node_path(name)
        os.makedirs(node_path, exist_ok=True)
        with open(node_path / 'error.txt', 'w') as f:
            json.dump(error_info, f, ensure_ascii=False, indent=2)

    def finalize(self, name):
        node_path = self._node_path(name)
        obj_file = node_path / 'obj.pkl'
        with open(obj_file, 'rb') as f:
            _, _, spec = pkl.load(f)
        with open(node_path / 'finalized.pkl', 'wb') as f:
            pkl.dump(spec, f)
        obj_file.unlink()

    def get_obj(self, name):
        with open(self._node_path(name) / 'obj.pkl', 'rb') as f:
            return pkl.load(f)

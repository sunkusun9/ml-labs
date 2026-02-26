import re


class Collector:
    """Base class for data collectors attached to an Experimenter.

    Subclasses override the lifecycle hooks ``_start``, ``_collect``,
    ``_end_idx``, and ``_end`` to capture data during :meth:`~mllabs.Experimenter.exp`.

    Args:
        name (str): Collector name (unique within an Experimenter).
        connector (Connector): Determines which Head nodes this collector
            attaches to.

    Attributes:
        path (Path | None): Set by Experimenter on registration.
    """

    def __init__(self, name, connector):
        self.name = name
        self.connector = connector
        self.path = None

    def _start(self, node):
        pass

    def _collect(self, node, idx, inner_idx, context):
        pass

    def _end_idx(self, node, idx):
        pass

    def _end(self, node):
        pass

    def has(self, node):
        return self.has_node(node)

    def has_node(self, node):
        return False

    def reset_nodes(self, nodes):
        pass

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

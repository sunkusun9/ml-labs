class Collector:
    def __init__(self, name, connector):
        self.name = name
        self.connector = connector
        self.path = None
        self.warnings = []
        self._node_paths = {}  # {node_name: collector_data_dir (node_path / self.name)}

    def collect(self, context):
        return None

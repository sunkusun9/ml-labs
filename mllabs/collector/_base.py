class Collector:
    def __init__(self, name, connector):
        self.name = name
        self.connector = connector
        self.path = None
        self.warnings = []
        self._buf = {}  # {(node, idx): [result, ...]}

    def collect(self, context):
        return None

    def push(self, node, idx, no, result):
        self._buf.setdefault((node, idx), []).append(result)

    def end_idx(self, node, idx):
        pass

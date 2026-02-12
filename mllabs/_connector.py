import re


class Connector:
    def __init__(self, node_query=None, edges=None, processor=None):
        self.node_query = node_query
        self.edges = edges
        self.processor = processor

    def match(self, node_name, node_attrs):
        if self.node_query is not None:
            if isinstance(self.node_query, str):
                if not re.search(self.node_query, node_name):
                    return False
            elif isinstance(self.node_query, list):
                if node_name not in self.node_query:
                    return False

        if self.processor is not None:
            if node_attrs.get('processor') != self.processor:
                return False

        if self.edges is not None:
            node_edges = node_attrs.get('edges', {})
            for key, required_edges in self.edges.items():
                if key not in node_edges:
                    return False
                for edge in required_edges:
                    if edge not in node_edges[key]:
                        return False

        return True

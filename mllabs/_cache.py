import sys
from cachetools import LRUCache


def _get_data_size(data):
    if data is None:
        return 0
    if isinstance(data, (list, tuple)):
        return sum(_get_data_size(item) for item in data)
    if hasattr(data, 'nbytes'):
        return data.nbytes
    if hasattr(data, 'memory_usage'):
        return data.memory_usage(deep=True).sum()
    return sys.getsizeof(data)


class DataCache:
    def __init__(self, maxsize=4 * 1024 ** 3):
        self.cache_dic = LRUCache(maxsize=maxsize, getsizeof=_get_data_size)

    def get_data(self, node, outer_idx, inner_idx, typ):
        return self.cache_dic.get((node, outer_idx, inner_idx, typ), None)

    def put_data(self, node, outer_idx, inner_idx, typ, data):
        self.cache_dic[(node, outer_idx, inner_idx, typ)] = data

    def clear(self):
        self.cache_dic.clear()

    def clear_nodes(self, nodes):
        node_set = set(nodes)
        for k in [k for k in self.cache_dic if k[0] in node_set]:
            del self.cache_dic[k]

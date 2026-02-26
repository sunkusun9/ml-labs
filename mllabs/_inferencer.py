import pickle as pkl
from pathlib import Path

from ._data_wrapper import wrap, unwrap
from ._node_processor import resolve_columns


class Inferencer:

    def __init__(self, pipeline, selected_stages, selected_heads, n_splits, node_objs, v=None):
        self.pipeline = pipeline
        self.selected_stages = selected_stages
        self.selected_heads = selected_heads
        self.n_splits = n_splits
        self.node_objs = node_objs
        self.v = v

    def process(self, data, agg='mean'):
        results = list(self._process_splits(data))

        if self.n_splits == 1:
            return unwrap(results[0])

        if agg is None:
            return [unwrap(i) for i in results]
        elif agg == 'mean':
            return unwrap(type(results[0]).mean(iter(results)))
        elif agg == 'mode':
            return unwrap(type(results[0]).mode(iter(results)))
        elif callable(agg):
            return unwrap(agg(results))
        else:
            raise ValueError(f"Unknown agg: {agg}")

    def _process_splits(self, data):
        data = wrap(data)
        ordered = [
            name for name in self.pipeline._get_affected_nodes([None])
            if name in set(self.selected_stages + self.selected_heads)
        ]
        stage_set = set(self.selected_stages)

        for split_idx in range(self.n_splits):
            data_dicts = {None: (data, None)}
            head_outputs = []

            for name in ordered:
                obj = self.node_objs[name][split_idx]
                node_attrs = self.pipeline.get_node_attrs(name)
                output = self._process_node(obj, data_dicts, node_attrs['edges'])
                if output is None:
                    continue
                if name in stage_set:
                    data_dicts[name] = (output, obj)
                else:
                    if self.v is not None:
                        cols = resolve_columns(output, self.v, processor=obj)
                        output = output.select_columns(cols)
                    head_outputs.append(output)

            if len(head_outputs) == 1:
                yield head_outputs[0]
            else:
                yield type(head_outputs[0]).concat(head_outputs, axis=1)

    def _process_node(self, obj, data_dicts, edges):
        input_data = self._get_process_data(data_dicts, edges)
        if input_data is None:
            return None
        return obj.process(input_data)

    def _get_process_data(self, data_dicts, edges):
        if 'X' not in edges:
            return None
        parts = []
        edge_list = edges['X']
        for src_node, var in edge_list:
            src, obj = data_dicts[src_node]
            if var is not None:
                cols = resolve_columns(src, var, processor=obj)
                src = src.select_columns(cols)
            parts.append(src)
        if len(parts) == 1:
            return parts[0]
        else:
            return type(parts[0]).concat(parts, axis=1)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_data = {
            'pipeline': self.pipeline,
            'selected_stages': self.selected_stages,
            'selected_heads': self.selected_heads,
            'n_splits': self.n_splits,
            'node_objs': self.node_objs,
            'v': self.v,
        }
        with open(path / '__inferencer.pkl', 'wb') as f:
            pkl.dump(save_data, f)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path / '__inferencer.pkl', 'rb') as f:
            save_data = pkl.load(f)
        return cls(
            save_data['pipeline'],
            save_data['selected_stages'],
            save_data['selected_heads'],
            save_data['n_splits'],
            save_data['node_objs'],
            save_data.get('v'),
        )

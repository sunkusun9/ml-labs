import pickle
import re
import shutil

import numpy as np
import pandas as pd
import shap

from ._base import Collector
from .._data_wrapper import unwrap


class SHAPCollector(Collector):
    _SAVE_EXCLUDE = {'_buf': dict}

    def __init__(self, name, connector, explainer_cls=None, data_filter=None):
        super().__init__(name, connector)
        self.explainer_cls = explainer_cls or shap.TreeExplainer
        self.data_filter = data_filter

    def collect(self, context):
        (train_X, _), valid_X = context['input']['X']

        model = context['processor'].obj
        explainer = self.explainer_cls(model)

        train_dict = {'X': train_X}
        if self.data_filter is not None:
            train_dict = self.data_filter(train_dict)

        valid_dict = {'X': valid_X}
        if self.data_filter is not None:
            valid_dict = self.data_filter(valid_dict)

        return {
            'train': explainer.shap_values(unwrap(train_dict['X'])),
            'valid': explainer.shap_values(unwrap(valid_dict['X'])),
            'train_index': train_dict['X'].get_index(),
            'valid_index': valid_dict['X'].get_index(),
            'columns': list(context['processor'].X_),
        }

    def push(self, node, outer_idx, inner_idx, result):
        node_dir = self.path / node
        node_dir.mkdir(parents=True, exist_ok=True)
        with open(node_dir / f'{outer_idx}_{inner_idx}.pkl', 'wb') as f:
            pickle.dump(result, f)

    def has_node(self, node):
        if self.path is None:
            return False
        p = self.path / node
        return p.is_dir() and any(p.glob('*.pkl'))

    def has(self, node):
        return self.has_node(node)

    def reset_nodes(self, nodes):
        for node in nodes:
            p = self.path / node
            if p.exists():
                shutil.rmtree(p)

    def _get_saved_nodes(self):
        if self.path is None:
            return []
        return [p.name for p in self.path.iterdir()
                if p.is_dir() and any(p.glob('*.pkl'))]

    def _get_nodes(self, nodes, available):
        if nodes is None:
            return available
        if isinstance(nodes, list):
            return [n for n in nodes if n in set(available)]
        return [n for n in available if re.search(nodes, n)]

    def _shap_to_importance(self, shap_vals, columns):
        if isinstance(shap_vals, list):
            abs_vals = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            abs_vals = np.abs(shap_vals)
        if abs_vals.ndim > 2:
            abs_vals = abs_vals.mean(axis=-1)
        return pd.Series(abs_vals.mean(axis=0), index=columns)

    def get_feature_importance(self, node, idx):
        p = self.path / node
        inner_files = sorted(
            (f for f in p.glob(f'{idx}_*.pkl')),
            key=lambda x: int(x.stem.split('_')[1])
        )
        result = []
        for f in inner_files:
            with open(f, 'rb') as fp:
                data = pickle.load(fp)
            inner_idx = int(f.stem.split('_')[1])
            result.append(self._shap_to_importance(data['valid'], data['columns']).rename(inner_idx))
        return result

    def get_feature_importance_agg(self, node, agg_inner='mean', agg_outer='mean'):
        p = self.path / node
        outer_idxs = sorted(set(int(f.stem.split('_')[0]) for f in p.glob('*.pkl')))

        outer_frames = []
        for idx in outer_idxs:
            inner_list = self.get_feature_importance(node, idx)
            df = pd.concat(inner_list, axis=1)
            if agg_inner is not None:
                df = df.agg(agg_inner, axis=1).rename(idx)
            else:
                df.columns = pd.MultiIndex.from_tuples([(idx, s.name) for s in inner_list])
            outer_frames.append(df)

        result = pd.concat(outer_frames, axis=1)
        if agg_outer is not None and agg_inner is not None:
            result = result.agg(agg_outer, axis=1)
        return result

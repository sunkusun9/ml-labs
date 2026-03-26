import numpy as np
import pandas as pd
import shap

from ._base import Collector
from .._data_wrapper import unwrap


class SHAPCollector(Collector):
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

    def _shap_to_importance(self, shap_vals, columns):
        if isinstance(shap_vals, list):
            abs_vals = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            abs_vals = np.abs(shap_vals)
        if abs_vals.ndim > 2:
            abs_vals = abs_vals.mean(axis=-1)
        return pd.Series(abs_vals.mean(axis=0), index=columns)

    def get_feature_importance(self, node, idx):
        data = self._load_collect_results(node)
        entries = sorted(
            ((k, v) for k, v in data.items() if k[0] == idx),
            key=lambda x: x[0][1]
        )
        return [
            self._shap_to_importance(v['valid'], v['columns']).rename(inner_idx)
            for (_, inner_idx), v in entries
        ]

    def get_feature_importance_agg(self, node, agg_inner='mean', agg_outer='mean'):
        data = self._load_collect_results(node)
        outer_indices = sorted(set(k[0] for k in data.keys()))

        outer_frames = []
        for idx in outer_indices:
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

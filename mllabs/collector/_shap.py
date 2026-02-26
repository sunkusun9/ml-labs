import pickle
import numpy as np
import pandas as pd
import shap

from ._base import Collector
from .._data_wrapper import unwrap


class SHAPCollector(Collector):
    """Computes SHAP values and feature importance for each fold.

    Applies an optional ``data_filter`` to subsample rows before computing
    SHAP values. Supports tree-based multiclass models (3-D SHAP arrays are
    averaged over the class axis).

    Args:
        name (str): Collector name.
        connector (Connector): Node matching criteria.
        explainer_cls: SHAP explainer class. Default ``shap.TreeExplainer``.
        data_filter (DataFilter, optional): Applied to train and valid data
            before calling the explainer.
    """

    def __init__(self, name, connector, explainer_cls=None, data_filter=None):
        super().__init__(name, connector)
        self.explainer_cls = explainer_cls or shap.TreeExplainer
        self.data_filter = data_filter
        self._buffer = {}
        self._mem_data = {}

    def _start(self, node):
        self._buffer[node] = {}

    def _collect(self, node, idx, inner_idx, context):
        (train_X, _), valid_X = context['input']['X']

        model = context['processor'].obj
        explainer = self.explainer_cls(model)

        train_dict = {'X': train_X}
        if self.data_filter is not None:
            train_dict = self.data_filter(train_dict)

        valid_dict = {'X': valid_X}
        if self.data_filter is not None:
            valid_dict = self.data_filter(valid_dict)

        self._buffer[node][(idx, inner_idx)] = {
            'train': explainer.shap_values(unwrap(train_dict['X'])),
            'valid': explainer.shap_values(unwrap(valid_dict['X'])),
            'train_index': train_dict['X'].get_index(),
            'valid_index': valid_dict['X'].get_index(),
            'columns': list(context['processor'].X_),
        }

    def _end_idx(self, node, idx):
        entries = {k: v for k, v in self._buffer[node].items() if k[0] == idx}
        if not entries:
            return

        if self.path is not None:
            self._ensure_path()
            file_path = self.path / f"{node}.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    existing = pickle.load(f)
                existing.update(entries)
            else:
                existing = dict(entries)
            with open(file_path, 'wb') as f:
                pickle.dump(existing, f)
        else:
            if node not in self._mem_data:
                self._mem_data[node] = {}
            self._mem_data[node].update(entries)

        for k in entries:
            del self._buffer[node][k]

    def _end(self, node):
        if node in self._buffer:
            del self._buffer[node]

    def has_node(self, node):
        if node in self._mem_data:
            return True
        if self.path is not None:
            return (self.path / f"{node}.pkl").exists()
        return False

    def reset_nodes(self, nodes):
        for node in nodes:
            if self.path is not None:
                file_path = self.path / f"{node}.pkl"
                if file_path.exists():
                    file_path.unlink()
            if node in self._mem_data:
                del self._mem_data[node]
            if node in self._buffer:
                del self._buffer[node]

    def save(self):
        if self.path is None:
            return
        self._ensure_path()
        config = {
            'name': self.name,
            'connector': self.connector,
            'explainer_cls': self.explainer_cls,
            'data_filter': self.data_filter,
        }
        with open(self.path / '__config.pkl', 'wb') as f:
            pickle.dump(config, f)

    @classmethod
    def load(cls, path):
        with open(path / '__config.pkl', 'rb') as f:
            config = pickle.load(f)
        obj = cls(
            name=config['name'],
            connector=config['connector'],
            explainer_cls=config.get('explainer_cls'),
            data_filter=config.get('data_filter'),
        )
        obj.path = path
        return obj

    def _load_node_data(self, node):
        if self.path is not None:
            file_path = self.path / f"{node}.pkl"
            if not file_path.exists():
                raise FileNotFoundError(f"SHAP data not found: {file_path}")
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            if node not in self._mem_data:
                raise KeyError(f"SHAP data not found in memory: {node}")
            return self._mem_data[node]

    def _get_saved_nodes(self):
        if self.path is not None:
            if not self.path.exists():
                return []
            return [f.stem for f in self.path.glob("*.pkl") if not f.stem.startswith('__')]
        else:
            return list(self._mem_data.keys())

    def _shap_to_importance(self, shap_vals, columns):
        if isinstance(shap_vals, list):
            abs_vals = np.mean([np.abs(sv) for sv in shap_vals], axis=0)
        else:
            abs_vals = np.abs(shap_vals)
        if abs_vals.ndim > 2:
            abs_vals = abs_vals.mean(axis=-1)
        return pd.Series(abs_vals.mean(axis=0), index=columns)

    def get_feature_importance(self, node, idx):
        """Return per-inner-fold feature importance for one outer fold.

        Args:
            node (str): Node name.
            idx (int): Outer fold index.

        Returns:
            list[pd.Series]: One Series per inner fold (mean absolute SHAP
            values over samples).
        """
        data = self._load_node_data(node)
        entries = sorted(
            ((k, v) for k, v in data.items() if k[0] == idx),
            key=lambda x: x[0][1]
        )
        return [
            self._shap_to_importance(v['valid'], v['columns']).rename(inner_idx)
            for (_, inner_idx), v in entries
        ]

    def get_feature_importance_agg(self, node, agg_inner='mean', agg_outer='mean'):
        """Return aggregated feature importance across all folds.

        Args:
            node (str): Node name.
            agg_inner (str | None): Aggregation function name for inner folds
                (passed to ``pd.DataFrame.agg``). ``None`` keeps inner fold
                axis as a MultiIndex level.
            agg_outer (str | None): Aggregation function name for outer folds.
                ``None`` returns a DataFrame with one column per outer fold.

        Returns:
            pd.Series | pd.DataFrame: When both *agg_inner* and *agg_outer* are
            set, returns a ``pd.Series``. When *agg_outer* is ``None``, returns
            a DataFrame. When *agg_inner* is ``None``, returns a MultiIndex DataFrame.
        """
        data = self._load_node_data(node)
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

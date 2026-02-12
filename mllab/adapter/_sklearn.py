from ._base import ModelAdapter

import pandas as pd
import numpy as np

class LMAdapter(ModelAdapter):
    @staticmethod
    def _get_coef( processor):
        coef_ = processor.obj.coef_
        if len(coef_.shape) == 1:
            if hasattr(processor.obj, 'intercept_'):
                coef_ = np.expand_dims(np.concatenate([coef_, [processor.obj.intercept_]]), axis=0)
                coef_name = list(processor.X_) + ['intercept']
            else:
                coef_ = np.expand_dims(processor.obj.coef_, axis=0)
                coef_name = processor.X_
            idx = [0]
        else:
            if hasattr(processor.obj, 'intercept_'):
                coef_ = np.concatenate([coef_, np.expand_dims(processor.obj.intercept_, axis=0)], axis=1)
                coef_name = list(processor.X_) + ['intercept']
            else:
                coef_ = processor.obj.coef_
                coef_name = processor.X_
            idx = np.arange(coef_.shape[0])
        return pd.DataFrame(coef_, index=idx, columns=coef_name).stack().rename('coef')

LMAdapter.result_objs = {'coef': (LMAdapter._get_coef, True)}

class PCAAdapter(ModelAdapter):
    @staticmethod
    def _get_explained_variance(processor):
        obj = processor.obj
        output_vars = list(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else list(range(len(obj.explained_variance_)))
        return pd.Series(
            obj.explained_variance_, index=output_vars, name = 'variance'
        )
    
    @staticmethod
    def _get_explained_variance_ratio(processor):
        obj = processor.obj
        output_vars = list(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else list(range(len(obj.explained_variance_ratio_)))
        return pd.Series(
            obj.explained_variance_ratio_, index=output_vars, name='ratio'
        )

    @staticmethod
    def _get_components(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.components_.shape[1]))
        output_vars = list(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else list(range(obj.components_.shape[0]))
        return pd.DataFrame(
            obj.components_,
            index=output_vars,
            columns=input_vars
        ).stack().rename('component')

PCAAdapter.result_objs = {
    'explained_variance': (PCAAdapter._get_explained_variance, True),
    'explained_variance_ratio': (PCAAdapter._get_explained_variance_ratio, True),
    'components': (PCAAdapter._get_components, True)
}


class LDAAdapter(ModelAdapter):
    @staticmethod
    def _get_coef(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.coef_.shape[-1]))
        classes = list(processor.classes_) if hasattr(processor, 'classes_') and processor.classes_ is not None else list(range(obj.coef_.shape[0] if obj.coef_.ndim > 1 else 1))

        coef = obj.coef_
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)

        return pd.DataFrame(
            coef,
            index=classes,
            columns=input_vars
        ).unstack().rename('value')

    @staticmethod
    def _get_intercept(processor):
        obj = processor.obj
        classes = list(processor.classes_) if hasattr(processor, 'classes_') and processor.classes_ is not None else list(range(len(np.atleast_1d(obj.intercept_))))

        intercept = obj.intercept_
        if np.isscalar(intercept):
            intercept = [intercept]

        return pd.Series(
            intercept, index=classes, name = 'intercept'
        )

    @staticmethod
    def _get_scalings(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.scalings_.shape[0]))
        output_vars = list(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else [f'LD{i}' for i in range(obj.scalings_.shape[1])]

        return pd.DataFrame(
            obj.scalings_,
            index=input_vars,
            columns=output_vars
        ).unstack().rename('scale')

    @staticmethod
    def _get_explained_variance_ratio(processor):
        obj = processor.obj
        output_vars = list(processor.output_vars) if hasattr(processor, 'output_vars') and processor.output_vars is not None else [f'LD{i}' for i in range(len(obj.explained_variance_ratio_))]

        return pd.Series(
            obj.explained_variance_ratio_, index=output_vars, name = 'ratio'
        )
LDAAdapter.result_objs = {
    'coef': (LDAAdapter._get_coef, True),
    'intercept': (LDAAdapter._get_intercept, True),
    'scalings': (LDAAdapter._get_scalings, True),
    'explained_variance_ratio': (LDAAdapter._get_explained_variance_ratio, True)
}

class DecisionTreeAdapter(ModelAdapter):
    @staticmethod
    def _get_feature_importances(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(len(obj.feature_importances_)))

        return pd.Series(
            obj.feature_importances_, index=input_vars, name = 'importance'
        )
    
    @staticmethod
    def _get_tree(processor):
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.n_features_in_))

        tree_ = obj.tree_
        tree_structure = []
        for i in range(tree_.node_count):
            feature_idx = tree_.feature[i]
            node_dict = {
                'node_id': i,
                'feature': input_vars[feature_idx] if feature_idx >= 0 else None,
                'threshold': tree_.threshold[i] if feature_idx >= 0 else None,
                'impurity': tree_.impurity[i],
                'n_samples': tree_.n_node_samples[i],
                'left_child': tree_.children_left[i] if tree_.children_left[i] >= 0 else None,
                'right_child': tree_.children_right[i] if tree_.children_right[i] >= 0 else None,
                'value': tree_.value[i].tolist()
            }
            tree_structure.append(node_dict)

        return pd.DataFrame(tree_structure)

    @staticmethod
    def _plot_tree(processor, **args):
        from sklearn.tree import plot_tree
        obj = processor.obj
        input_vars = list(processor.X_) if hasattr(processor, 'X_') and processor.X_ is not None else list(range(obj.n_features_in_))
        plot_tree(obj, feature_names = input_vars, **args)

DecisionTreeAdapter.result_objs = {
    'feature_importances': (DecisionTreeAdapter._get_feature_importances, True),
    'tree': (DecisionTreeAdapter._get_tree, False), 
    'plot_tree': (DecisionTreeAdapter._plot_tree, False), 
}
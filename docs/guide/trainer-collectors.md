# Trainer & Collectors

## Trainer

### Cross-Validation Workflow

`Trainer` runs cross-validation training independently of the main `Experimenter` experiment loop. It is created through `Experimenter.add_trainer()` and shares the same Pipeline and data cache.

```python
trainer = exp.add_trainer(
    name='cv',
    data=None,              # None → use Experimenter's data
    splitter='same',        # 'same' → use exp.sp_v (inner splitter)
    splitter_params=None,   # None when splitter='same'
)
```

`splitter='same'` reuses the inner splitter (`sp_v`) configured on the Experimenter. Pass a scikit-learn splitter object to use a different split strategy. `splitter=None` trains on the entire dataset without splitting.

### select_head, train, process

**`select_head(nodes)`** specifies which Head nodes to train. All upstream Stage nodes are collected automatically.

```python
trainer.select_head(['lgbm_v1', 'lgbm_v2'])
```

**`train()`** trains all unbuilt nodes in topological order. Each node is trained across all splits before moving to the next node.

```python
trainer.train()
```

**`process(data, v=None)`** is a generator that applies the trained processors to new data, yielding one result per split.

```python
for split_output in trainer.process(test_df):
    # split_output: concatenated Head outputs for this split
    ...
```

`v` optionally filters output columns from the Head nodes. If multiple Heads are selected, their outputs are concatenated column-wise.

### to_inferencer

Once training is complete, convert the Trainer to an `Inferencer` for deployment:

```python
inferencer = trainer.to_inferencer(v=None)
inferencer.save('./inferencer')
```

`to_inferencer()` copies the fitted processors out of the Trainer, so the resulting `Inferencer` is independent of the Trainer and Experimenter.

---

## Collectors

Collectors capture data from Head nodes during `exp()`. Each Collector uses a `Connector` to select which nodes it observes.

### Connector-Based Matching

`Connector` controls which nodes a Collector attaches to. All three criteria are optional — only the ones provided are checked:

```python
from mllabs import Connector

Connector()                                  # matches all nodes
Connector(node_query='lgbm')                 # regex match on node name
Connector(node_query=['lgbm_v1', 'lgbm_v2']) # exact list match
Connector(processor=LGBMClassifier)          # processor class match
Connector(edges={'y': [(None, 'target')]})   # edges contain-based match
```

Multiple criteria are combined with AND logic.

---

### MetricCollector

Computes a metric function against ground truth `y` for each fold.

```python
from mllabs.collector import MetricCollector
from sklearn.metrics import log_loss

mc = MetricCollector(
    name='metrics',
    connector=Connector(),
    output_var=None,          # None → all output columns
    metric_func=log_loss,     # func(y_true, y_pred) → scalar
    include_train=True,       # also compute on train/inner-valid folds
)
exp.add_collector(mc)
```

**Querying results:**

```python
mc = exp.get_collector('metrics')

mc.get_metric('lgbm_v1')          # Series of per-fold metrics
mc.get_metrics(['lgbm_v1', 'lgbm_v2'])   # DataFrame

# Aggregate across folds
mean, std = mc.get_metrics_agg(
    nodes=None,         # None → all collected nodes
    inner_fold=True,    # aggregate inner folds (mean)
    outer_fold=True,    # then aggregate outer folds (mean)
    include_std=True,   # also return std DataFrame
)
```

---

### StackingCollector

Collects out-of-fold (OOF) predictions for stacking.

```python
from mllabs.collector import StackingCollector

sc = StackingCollector(
    name='stacking',
    connector=Connector(edges={'y': [(None, 'target')]}),
    output_var=None,          # columns to collect from output
    experimenter=exp,         # used to build index and target
    method='mean',            # how to aggregate inner folds: 'mean', 'mode', 'simple'
)
exp.add_collector(sc)
```

The `connector.edges` `'y'` entry is used to extract the target column into the dataset.

**Querying results:**

```python
sc = exp.get_collector('stacking')

df = sc.get_dataset(
    nodes=None,           # None → all collected nodes
    include_target=True,  # append target column
)
# Returns a DataFrame with OOF predictions + target, indexed to match original data
```

---

### ModelAttrCollector

Collects model attributes such as feature importances for each fold.

```python
from mllabs.collector import ModelAttrCollector

mac = ModelAttrCollector(
    name='importance',
    connector=Connector(processor=LGBMClassifier),
    result_key='feature_importances',  # key in adapter.result_objs
    # adapter is inferred from connector.processor automatically
)
exp.add_collector(mac)
```

**Querying results:**

```python
mac = exp.get_collector('importance')

mac.get_attr('lgbm_v1')           # raw results: list of outer folds, each a list of inner folds
mac.get_attr('lgbm_v1', idx=0)    # results for outer fold 0

# Aggregate (only for mergeable result types like feature importances)
series = mac.get_attrs_agg(
    node='lgbm_v1',
    agg_inner=True,   # mean across inner folds
    agg_outer=True,   # then mean across outer folds → returns Series
)
# agg_inner=True, agg_outer=False → returns DataFrame (one column per outer fold)
# agg_inner=False → raises ValueError
```

---

### SHAPCollector

Computes SHAP values using a tree explainer for each fold.

```python
from mllabs.collector import SHAPCollector
from mllabs.filter import RandomFilter

shap_c = SHAPCollector(
    name='shap',
    connector=Connector(processor=LGBMClassifier),
    explainer_cls=None,       # None → shap.TreeExplainer
    data_filter=RandomFilter(n=500, random_state=0),  # subsample for speed
)
exp.add_collector(shap_c)
```

`data_filter` is applied to both train and valid data before computing SHAP values.

**Querying results:**

```python
shap_c = exp.get_collector('shap')

# Per outer fold: list of pd.Series (one per inner fold)
series_list = shap_c.get_feature_importance('lgbm_v1', idx=0)

# Aggregated across all folds
importance = shap_c.get_feature_importance_agg(
    node='lgbm_v1',
    agg_inner='mean',   # aggregate inner folds; None → keep MultiIndex
    agg_outer='mean',   # aggregate outer folds; None → return DataFrame
)
# Both set → Series; agg_outer=None → DataFrame; agg_inner=None → MultiIndex DataFrame
```

Multiclass SHAP arrays `(n_samples, n_features, n_classes)` are automatically averaged across the class axis before computing feature importance.

---

### OutputCollector

Saves raw `output_train` and `output_valid` arrays to disk for each fold.

```python
from mllabs.collector import OutputCollector

oc = OutputCollector(
    name='outputs',
    connector=Connector(),
    output_var=None,        # columns to capture
    include_target=True,
)
exp.add_collector(oc)
```

**Querying results:**

```python
oc = exp.get_collector('outputs')

entry = oc.get_output('lgbm_v1', idx=0, inner_idx=0)
# entry: {'output_train': (train_arr, valid_sub_arr), 'output_valid': arr, 'columns': [...]}

all_entries = oc.get_outputs('lgbm_v1')
# {(idx, inner_idx): entry, ...}
```
